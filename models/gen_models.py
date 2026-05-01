import copy
import math
import numpy as np
import torch
from tqdm import tqdm


def initialize_gen_model(args, device):
    if args.task == 'dna':
        from Enformer import initialize_gen_model_dna
        pre_model = initialize_gen_model_dna(
            task=args.task,
            base_path=args.data_base_path,
            device=device,
            grad=False,
        )  # no update
        old_model = initialize_gen_model_dna(
            task=args.task,
            base_path=args.data_base_path,
            device=device,
            grad=False,
        )  # only copy cur_model params
        new_model = initialize_gen_model_dna(
            task=args.task,
            base_path=args.data_base_path,
            device=device,
            grad=True,
            pretrained=True if args.student_initialize_pretrain else False,  # random initialize params
        )
        tokenizer = None
    elif args.task == 'protein':
        from models.protein_gen_models import ProteinGenDiffusion
        # initialize evodiff

        pre_model = ProteinGenDiffusion(args).to(device)
        old_model = ProteinGenDiffusion(args).to(device)
        new_model = ProteinGenDiffusion(args).to(device)

        pre_model.eval()
        for param in pre_model.parameters():
            param.requires_grad = False
        old_model.eval()
        for param in old_model.parameters():
            param.requires_grad = False
        tokenizer = None
    else:
        raise NotImplementedError()
    model_collections = {
        "pre_model": pre_model,
        "old_model": old_model,
        "new_model": new_model,
        "tokenizer": tokenizer,
    }
    return model_collections


def generate_xt_list(
        args,
        generative_model,
        device,
        timesteps=0,
        dt=0,
        seq_len=0,
        unmask_K=1,
        reward_model=None,
        make_svdd=False,
        num_best_of_N=1,
        train_stage=False,
        frozen_template_tokens=None,
        cdr_indices=None,
):
    """
    generate a list of xt, roll in distribution

    For task='ab', ``frozen_template_tokens`` is a 1-D LongTensor of length
    ``seq_len`` containing framework token IDs at non-CDR positions and the
    tokenizer mask_id at CDR positions. ``cdr_indices`` is the list of 0-based
    CDR positions (the only positions the sampler is allowed to unmask).
    """
    if args.task == 'dna':
        if num_best_of_N > 1:
            sample, last_x_list, condt_list, conds_list, move_chance_t_list, move_chance_s_list, copy_flag_list = \
                generative_model.sample_policy_distillation_best_of_n(
                    num_steps=args.total_num_steps,
                    batch_size=args.batch_size,
                    timesteps=timesteps,
                    dt=dt,
                    reward_model=reward_model,
                    eps=args.eps,
                    decode_alg='SVDDtw' if make_svdd else args.decode_alg,
                    sample_M=args.sample_M,
                    num_best_of_N=num_best_of_N,
                )
        else:
            sample, last_x_list, condt_list, conds_list, move_chance_t_list, move_chance_s_list, copy_flag_list = \
                generative_model.sample_policy_distillation(
                    num_steps=args.total_num_steps,
                    batch_size=args.batch_size,
                    timesteps=timesteps,
                    dt=dt,
                    reward_model=reward_model,
                    eps=args.eps,
                    decode_alg='SVDDtw' if make_svdd else args.decode_alg,
                    sample_M=args.sample_M,
                )
        # sample=[bsz, seqlen, 4] (one-hot but with grad),
        # last_x_list, len=128, each shape= [32, 200, 5]
        # condt_list, len=128, each shape= [32,]
        # move_chance_t_list, len=128, each shape= [32,]
        # copy_flag_list, len=128, each shape= [32,200,1]
    elif args.task in ('protein', 'ab'):
        ab_mode = args.task == 'ab'
        if ab_mode:
            if frozen_template_tokens is None:
                raise ValueError("task='ab' requires frozen_template_tokens (framework AA tokens with mask at CDR positions).")
            if not cdr_indices:
                raise ValueError("task='ab' requires cdr_indices (the only positions the sampler is allowed to unmask).")
            template_row = frozen_template_tokens.to(device).to(torch.long)
            if template_row.dim() != 1 or template_row.shape[0] != seq_len:
                raise ValueError(
                    f"frozen_template_tokens shape {tuple(template_row.shape)} != (seq_len={seq_len},)"
                )
            cdr_set = set(int(i) for i in cdr_indices)
            num_unmask_steps = math.ceil(len(cdr_set) / unmask_K)
        else:
            template_row = None
            cdr_set = None
            num_unmask_steps = math.ceil(seq_len / unmask_K)

        final_reward_list = []
        final_samples_list = []
        final_last_x_list = []  # N, timestep, bs*len
        final_condt_list = []
        # final_conds_list = []
        final_move_chance_t_list = []
        # final_move_chance_s_list = []
        final_copy_flag_list = []
        for n in range(num_best_of_N):
            last_x_list = []
            condt_list, conds_list = [], []
            move_chance_t_list, move_chance_s_list = [], []
            copy_flag_list = []

            mask = generative_model.tokenizer.mask_id
            pad = generative_model.tokenizer.pad_id

            if ab_mode:
                # Framework frozen, CDR positions masked. xt_to_xs only ever
                # rewrites positions present in loc_set, and copy_flag is
                # derived from (sample != mask_id) so framework tokens are
                # preserved automatically across the reverse process.
                sample = template_row.unsqueeze(0).expand(args.batch_size, -1).contiguous()
                loc_set = [set(cdr_set) for _ in range(args.batch_size)]
            else:
                # Whole-sequence design: start from all-mask.
                sample = (torch.zeros((args.batch_size, seq_len)) + mask).to(torch.long).to(device)
                loc = np.arange(seq_len)
                loc_set = [set(loc) for _ in range(args.batch_size)]

            timestep = torch.tensor([0] * args.batch_size).to(device)  # placeholder but not called in model

            for i in tqdm(range(num_unmask_steps), desc="sequence generation", disable=train_stage):
            # for i in range(args.total_num_steps):
                if args.decode_alg == 'SVDDtw' or make_svdd:
                    _, sample_fake, loc_set_fake, copy_flag = generative_model.xt_to_xs_svdd(sample, timestep, loc_set, reward_model, unmask_K=unmask_K, num_candidate=args.SVDD_num_candidate)
                elif args.decode_alg == 'sampling':
                    _, sample_fake, loc_set_fake, copy_flag = generative_model.xt_to_xs(sample, timestep, loc_set, unmask_K=unmask_K)
                else:
                    raise NotImplementedError()

                last_x_list.append(sample.detach())
                condt_list.append(timestep)
                move_chance_t_list.append(loc_set)
                copy_flag_list.append(copy_flag)
                conds_list.append(0)
                move_chance_s_list.append(0)

                sample = sample_fake
                loc_set = loc_set_fake

            last_x_list = torch.stack(last_x_list, dim=0)  # timestep, bs, len
            condt_list = torch.stack(condt_list, dim=0)  # timestep, bs
            # conds_list = torch.tensor(conds_list, dtype=torch.int, device=device)  # timestep
            move_chance_t_list = move_chance_t_list  # timestep * bs, set
            # move_chance_s_list = torch.tensor(move_chance_s_list, dtype=torch.int, device=device)  # timestep
            copy_flag_list = torch.stack(copy_flag_list, dim=0)  # timestep, bs, len

            # get reward
            cur_reward_list = reward_model.reward_metrics(S_sp=sample, save_pdb=False) if num_best_of_N > 1 else [0 for i in range(args.batch_size)]  # [bs]

            final_reward_list.append(cur_reward_list)
            final_samples_list.append(sample.clone())
            final_last_x_list.append(last_x_list)
            final_condt_list.append(condt_list)
            # final_conds_list.append(conds_list)
            final_move_chance_t_list.append(move_chance_t_list)
            # final_move_chance_s_list.append(move_chance_s_list)
            final_copy_flag_list.append(copy_flag_list)

        # best of N
        reward_tensor = np.stack(final_reward_list)  # [N, bs]
        best_indices = np.argmax(reward_tensor, axis=0)  # [bs]

        final_samples_list = torch.stack(final_samples_list, dim=0)  # N, bs, len
        final_last_x_list = torch.stack(final_last_x_list, dim=0).permute(0,2,1,3)  # N, bs, timestep, len
        final_condt_list = torch.stack(final_condt_list, dim=0).permute(0,2,1)  # N, bs, timestep
        # final_conds_list = torch.stack(final_conds_list, dim=0)  # N, timestep
        final_move_chance_t_list = final_move_chance_t_list  # N * timestep * bs, set
        # final_move_chance_s_list = torch.stack(final_move_chance_s_list, dim=0)  # N, timestep
        final_copy_flag_list = torch.stack(final_copy_flag_list, dim=0).permute(0,2,1,3)  # N, bs, timestep, len

        bs_indices = torch.arange(args.batch_size)
        best_sample = final_samples_list[best_indices, bs_indices, :]  # bs*len
        best_last_x_list = final_last_x_list[best_indices, bs_indices, :, :].permute(1,0,2)  # timestep, bs, len
        best_condt_list = final_condt_list[best_indices, bs_indices, :].permute(1,0)  # timestep, bs
        best_copy_flag_list = final_copy_flag_list[best_indices, bs_indices, :, :].permute(1,0,2)  # timestep, bs, len

        total_T = num_unmask_steps
        best_move_chance_t_list = [[] for _ in range(total_T)]
        for bs_idx in range(args.batch_size):
            best_n = best_indices[bs_idx]
            for t in range(total_T):  # timestep
                best_move_chance_t_list[t].append(final_move_chance_t_list[best_n][t][bs_idx])

        sample = best_sample
        last_x_list = best_last_x_list
        condt_list = best_condt_list
        move_chance_t_list = best_move_chance_t_list
        copy_flag_list = best_copy_flag_list

    else:
        raise NotImplementedError()

    return sample, last_x_list, condt_list, conds_list, move_chance_t_list, move_chance_s_list, copy_flag_list
