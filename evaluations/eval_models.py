from evaluations.dna_eval import DNAEvalMetrics


def initialize_eval_model(args, device, result_save_folder=""):
    if args.task == 'dna':
        eval_models = DNAEvalMetrics(
            base_path=args.data_base_path,
            device=device,
        )
    elif args.task == 'protein':
        if 'iptm' in args.reward:
            from evaluations.protein_eval_bind_colabdesign import ProteinEvalMetricsColabDesign
            eval_models = ProteinEvalMetricsColabDesign(args, device, result_save_folder=result_save_folder)
        else:
            from evaluations.protein_eval import ProteinEvalMetrics
            eval_models = ProteinEvalMetrics(
                args=args,
                device=device,
                result_save_folder=result_save_folder,
            )
    elif args.task == 'ab':
        from evaluations.ab_af2_reward import AbAF2RewardCal
        eval_models = AbAF2RewardCal(args, device, result_save_folder=result_save_folder)
    else:
        raise NotImplementedError(f"Unknown --task '{args.task}'")

    return eval_models


