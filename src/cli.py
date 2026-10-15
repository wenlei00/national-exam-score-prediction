"""Research stages are explicit; prediction never starts a refit implicitly."""
import argparse
from pathlib import Path
from settings import ROOT

def main():
    parser = argparse.ArgumentParser(description='国考最低进面分研究：默认冻结训练截至2025年，2026年仅检验')
    parser.add_argument('stage', choices=['audit', 'build-panel', 'prepare-target', 'train-frozen',
                                         'evaluate-2026', 'predict-2027', 'refit-2027', 'report'])
    parser.add_argument('--include-2026-training', action='store_true', help='仅用于明确选择重训对照实验')
    args = parser.parse_args()
    if args.stage == 'refit-2027' and not args.include_2026_training:
        parser.error('对照实验需要显式传入 --include-2026-training；主版本使用 predict-2027')
    if args.include_2026_training and args.stage != 'refit-2027':
        parser.error('--include-2026-training 仅适用于 refit-2027')
    if args.stage == 'train-frozen' and (ROOT/'results/models/final_model_2019_2025.joblib').exists():
        parser.error('冻结模型已存在。请在新的独立项目副本中重训，避免覆盖检验依据。')
    modules = {'audit': ('inventory','inspect'), 'build-panel': ('data_processing','process'),
               'prepare-target': ('prepare_target','prepare'), 'train-frozen': ('train','run'),
               'evaluate-2026': ('evaluate_2026','run'), 'predict-2027': ('predict_2027_frozen','run'),
               'refit-2027': ('predict_2027','run'), 'report': ('report','run')}
    import importlib
    module, function = modules[args.stage]
    (ROOT/'data/processed').mkdir(parents=True, exist_ok=True)
    (ROOT/'results/plots').mkdir(parents=True, exist_ok=True)
    getattr(importlib.import_module(module), function)()

if __name__ == '__main__':
    main()
