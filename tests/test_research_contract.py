"""Synthetic-only tests of identifiers and temporal leakage boundaries."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from data_processing import JOB_FIELDS, identifier
from features import build_features, ALL_FEATURES, HISTORY_NUM
from settings import default_config, ROOT, resolve_path
from inventory import inputs

def synthetic_panel():
    rows = []
    for year, label in [(2019, 110), (2020, 120), (2021, 130), (2025, 140), (2026, 150), (2027, None)]:
        row = {c: '' for c in JOB_FIELDS}
        row.update({'部门代码':'001000','职位代码':'100110001001','部门名称':'示例部门','用人司局':'示例司局',
                    '工作地点':'浙江省杭州市','考试类别':'行政执法类','职位属性':'普通职位','专业':'统计学',
                    '招考人数':'2','面试人员比例':'3:1','year':year,
                    'min_interview_score':label,'train_eligible':year<2027})
        rows.append(row)
    return pd.DataFrame(rows)

class ResearchContract(unittest.TestCase):
    def test_identifier_zero_padding(self):
        self.assertEqual(identifier('1000',6), '001000')
        self.assertEqual(identifier('1000.0',6), '001000')

    def test_identifier_does_not_truncate(self):
        self.assertEqual(identifier('1234567',6), '1234567')

    def test_same_year_label_does_not_enter_features(self):
        p=synthetic_panel(); a=build_features(p)
        p.loc[p.year==2026,'min_interview_score']=999
        b=build_features(p)
        pd.testing.assert_frame_equal(a.loc[a.year==2026,ALL_FEATURES],b.loc[b.year==2026,ALL_FEATURES])

    def test_future_label_does_not_enter_past_features(self):
        p=synthetic_panel(); a=build_features(p)
        p.loc[p.year>=2025,'min_interview_score']=999
        b=build_features(p)
        pd.testing.assert_frame_equal(a.loc[a.year<=2021,ALL_FEATURES],b.loc[b.year<=2021,ALL_FEATURES])

    def test_history_is_limited_to_three_years(self):
        f=build_features(synthetic_panel())
        row=f.loc[f.year==2027].iloc[0]
        self.assertEqual(row['department_history_mean'],145)
        self.assertEqual(row['department_history_n'],2)

    def test_first_year_does_not_borrow_future(self):
        f=build_features(synthetic_panel())
        self.assertTrue(f.loc[f.year==2019,HISTORY_NUM[::2]].isna().all().all())

    def test_unlabeled_target_cannot_enter_history(self):
        p=synthetic_panel(); p.loc[p.year==2026,'train_eligible']=False
        f=build_features(p)
        self.assertEqual(f.loc[f.year==2027,'department_history_mean'].iloc[0],140)

    def test_frozen_configuration_rejects_2026_labels(self):
        cfg=default_config(); cfg['historical_inputs']=[dict(year=2026,kind='scores',path='fake.xlsx')]
        with patch('inventory.config',return_value=cfg), self.assertRaises(ValueError): inputs()

    def test_example_configuration_is_portable(self):
        cfg=default_config()
        self.assertTrue(all(not Path(x['path']).is_absolute() for x in cfg['historical_inputs']))
        self.assertEqual(resolve_path('data/raw/example.xls'),ROOT/'data/raw/example.xls')

if __name__=='__main__': unittest.main()
