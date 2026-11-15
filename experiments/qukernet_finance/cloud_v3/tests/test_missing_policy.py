"""Constructed-data verification of the versioned missing policy; no cloud I/O."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

from cloud_v3.features_missing import (build_features, _slope, _r2, _moment,
                                      _drawdown, _weighted, _corr, FEATURE_DICTIONARY)
from cloud_v3.data import (FEATURES, fit_missing_preprocessing, apply_missing_preprocessing,
                          digest, prepare_missing_window, calendar_split, select_universe,
                          _meta, MISSING_PROTOCOL_ID, _balanced_with_audit, save_window, load_window)


def bars(n=160):
    dates = pd.bdate_range("2025-01-02", periods=n)
    rng = np.random.default_rng(512)
    close = 10*np.exp(np.cumsum(rng.normal(0,.01,n)))
    return pd.DataFrame({"symbol":"sh600000", "date":dates, "open":close*.998,
                         "high":close*1.01, "low":close*.99, "close":close,
                         "volume":rng.integers(100,1000,n).astype(float)}), dates


class MissingOperatorTests(unittest.TestCase):
    def test_fixed_calendar_keeps_ref_gap_but_rolling_skips_missing(self):
        raw, dates = bars()
        missing = raw.drop(index=130)
        result = build_features(missing, dates)
        self.assertTrue(np.isnan(result.log_momentum_1.iloc[131]))
        expected = raw.close.iloc[127:132].drop(index=130).mean()
        self.assertAlmostEqual(result.mean_deviation_5.iloc[131], raw.close.iloc[131]/expected-1)
        self.assertFalse(result.history_complete.iloc[131])
        self.assertTrue(result.row_eligible.iloc[131])
        self.assertEqual(result.support__mean_deviation_5.iloc[131], 4)
        self.assertFalse(result.bar_valid.iloc[130])

    def test_all_missing_and_zero_volume_are_not_fabricated_statistics(self):
        raw, dates = bars()
        result = build_features(raw.drop(index=range(120,145)), dates)
        self.assertTrue(np.isnan(result.zero_volume_fraction_20.iloc[143]))
        self.assertTrue(np.isnan(result.path_efficiency_20.iloc[143]))
        raw.loc[120:139,"volume"] = 0
        zero = build_features(raw, dates)
        self.assertEqual(zero.zero_volume_fraction_20.iloc[139], 1.)
        self.assertTrue(np.isnan(zero.signed_volume_20.iloc[139]))

    def test_boolean_missing_is_excluded_from_denominator(self):
        raw, dates = bars()
        raw.loc[120:139,"volume"] = 0
        result = build_features(raw.drop(index=130), dates)
        self.assertEqual(result.zero_volume_fraction_20.iloc[139], 1.)
        self.assertEqual(result.support__zero_volume_fraction_20.iloc[139], 19)

    def test_weighted_statistics_use_identical_support(self):
        value = pd.Series([1., np.nan, 3., np.nan])
        weight = pd.Series([1.,1000.,1.,np.nan])
        self.assertEqual(_weighted(value,weight,3).iloc[2],2.)
        self.assertTrue(np.isnan(_weighted(pd.Series([np.nan]),pd.Series([2.]),1).iloc[0]))
        self.assertTrue(np.isnan(_weighted(pd.Series([1.]),pd.Series([0.]),1).iloc[0]))

    def test_pairwise_corr_discards_one_sided_values_and_joint_constant(self):
        a=pd.Series([1.,1.,9.,np.nan]); b=pd.Series([2.,3.,np.nan,100.])
        self.assertTrue(np.isnan(_corr(a,b,4).iloc[-1]))
        self.assertAlmostEqual(_corr(pd.Series([1.,np.nan,3.]),pd.Series([2.,100.,6.]),3).iloc[-1],1.)
        self.assertTrue(np.isnan(_corr(pd.Series([1.]),pd.Series([2.]),1).iloc[0]))

    def test_custom_statistics_keep_original_slope_positions(self):
        self.assertAlmostEqual(_slope([1.,np.nan,5.]),2.)
        self.assertEqual(_r2([1.,np.nan,5.]),1.)
        self.assertTrue(np.isnan(_slope([np.nan,1.])))
        self.assertTrue(np.isnan(_r2([2.,2.])))
        self.assertTrue(np.isnan(_moment([1.],3)))
        self.assertEqual(_moment([1.,np.nan,3.],3),0.)
        self.assertEqual(_moment([1.,np.nan,3.],4),-2.)
        self.assertEqual(_drawdown([3.]),0.)
        self.assertTrue(np.isnan(_drawdown([np.nan])))
        self.assertAlmostEqual(_drawdown([10.,np.nan,8.,9.]),.2)

    def test_illiquidity_coverage_keeps_eighty_percent_rule(self):
        raw, dates = bars()
        one_gap=build_features(raw.drop(index=130),dates)
        self.assertTrue(np.isfinite(one_gap.illiquidity_20.iloc[139]))
        many=build_features(raw.drop(index=range(125,130)),dates)
        self.assertTrue(np.isnan(many.illiquidity_20.iloc[139]))

    def test_label_endpoint_never_imputed(self):
        raw,dates=bars()
        result=build_features(raw.drop(index=131),dates)
        self.assertFalse(result.target_valid.iloc[130])
        self.assertTrue(np.isnan(result.y_raw.iloc[130]))

    def test_policy_dictionary_has_no_old_completeness_claim(self):
        edges=" ".join(r["edge_policy"] for r in FEATURE_DICTIONARY)
        self.assertNotIn("missing price/volume invalidates window",edges)
        self.assertIn("Empirical central moments",edges)
        self.assertIn("Observed finite closing-price path",edges)


class MissingPreprocessingTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(84)
        self.x=rng.normal(size=(160,80));self.y=np.arange(160)%2
        self.x[::7,3]=np.nan

    def test_training_median_exclusions_and_eight_features(self):
        self.x[:,0]=np.nan;self.x[:,1]=3.;self.x[:,2]=self.x[:,4]
        self.x[:,5]=0.;self.x[0,5]=1.
        fitted=fit_missing_preprocessing(self.x,self.y)
        removed={r["index"]:r["reason"] for r in fitted["imputer"]["excluded_features"]}
        self.assertEqual(removed[0],"all_missing_on_full_T")
        self.assertEqual(removed[1],"constant_after_training_median")
        self.assertEqual(removed[4],"exact_duplicate_after_training_median")
        self.assertEqual(removed[5],"zero_iqr_after_training_median")
        self.assertEqual(len(fitted["selected_indices"]),8)
        self.assertEqual(fitted["imputer"]["finite_counts"][3],137)
        self.assertEqual(fitted["scaler"]["fit_rows"],160)
        out=apply_missing_preprocessing(self.x,fitted)
        self.assertTrue(np.isfinite(out["angles"]).all())
        self.assertEqual(out["missing_mask80"].shape,(160,80))

    def test_validation_and_outer_cannot_refit_medians_or_selection(self):
        fitted=fit_missing_preprocessing(self.x,self.y)
        before=digest(fitted)
        a=np.full((50,80),np.nan)
        apply_missing_preprocessing(a,fitted)
        apply_missing_preprocessing(np.full((50,80),1e8),fitted)
        self.assertEqual(digest(fitted),before)
        self.assertEqual(fitted,fit_missing_preprocessing(self.x,self.y))

    def test_missing_selected_values_transform_to_training_center(self):
        fitted=fit_missing_preprocessing(self.x,self.y)
        out=apply_missing_preprocessing(np.full((16,80),np.nan),fitted)
        np.testing.assert_allclose(out["angles"],0.,atol=1e-14)
        self.assertTrue(out["missing_mask"].all())

    def test_under_eight_valid_columns_stops(self):
        x=np.full((160,80),np.nan);x[:,:7]=self.x[:,:7]
        with self.assertRaisesRegex(ValueError,"Fewer than eight"):
            fit_missing_preprocessing(x,self.y)


class MissingWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from cloud_v3.tests.test_data import constructed_features
        cls.dates = pd.bdate_range(end="2026-03-25", periods=500)
        records = [{"symbol":f"sh60{i:04d}","exchange":"sh",
                    "observed_valid_dates":cls.dates.strftime("%Y-%m-%d").tolist()} for i in range(60)]
        cls.manifest = {"latest_date":"2026-03-25","securities":records,
                        "calendar_provenance":{"status":"constructed_test_only","evidence":"unit test"}}
        symbols,universe,_,cls.split=select_universe(cls.manifest,cls.dates,latest_date="2026-03-25")
        cls.raw = pd.DataFrame([{"symbol":s,"date":d,"open":10.,"high":12.,"low":9.,"close":11.,"volume":1000.}
                                for s in symbols for d in cls.dates])
        cls.frame=constructed_features(cls.raw,cls.dates)
        cls.frame["row_eligible"]=True;cls.frame["history_span"]=121
        support=pd.DataFrame({"support__"+name:np.full(len(cls.frame),20,dtype=np.int16) for name in FEATURES})
        cls.frame=pd.concat([cls.frame,support],axis=1)
        frozen={"schema":"cloud_v3_frozen_sample_keys_v1","protocol_id":MISSING_PROTOCOL_ID,
                "universe":universe,"selected_source_manifest":cls.manifest,"sampling":{}}
        for phase,seed in (("train",20261115),("valid",20261116)):
            rows,trace=_balanced_with_audit(cls.frame[cls.frame.j.isin(cls.split[phase+"_days"])],
                                          cls.split[phase+"_days"],cls.dates,seed,phase)
            frozen[phase]=_meta(rows,cls.dates);frozen["sampling"][phase]=trace
            if phase=="train":cls.frame.loc[rows.index,"y_raw"]=np.array([-.01,.01]*8)
        frozen["sha256"]=digest(frozen);cls.frozen=frozen
        cls.protocol=json.loads((Path(__file__).parents[1]/"protocol_cutoff_20260325_qlib_missing_v1.json").read_text("utf-8"))

    def prepare(self, frame=None, frozen=None):
        with patch("cloud_v3.features_missing.build_features",return_value=self.frame if frame is None else frame):
            return prepare_missing_window(self.raw,self.dates,self.manifest,latest_date="2026-03-25",
                                          protocol=self.protocol,frozen_samples=self.frozen if frozen is None else frozen)

    def test_frozen_samples_and_snapshot_masks_roundtrip(self):
        import tempfile
        window=self.prepare()
        self.assertEqual(window["meta_train"],self.frozen["train"])
        self.assertEqual(window["meta_valid"],self.frozen["valid"])
        self.assertEqual(window["schema_version"],"cloud_v3_qlib_missing_data_v1")
        self.assertEqual(window["missing_mask_test"].shape,(50,8))
        self.assertEqual(window["support_counts_test"].shape,(50,80))
        with tempfile.TemporaryDirectory() as temp:
            receipt=save_window(window,Path(temp)/"private",source_manifest=self.manifest)
            restored=load_window(receipt["npz"])
        np.testing.assert_array_equal(restored["x_test"],window["x_test"])
        np.testing.assert_array_equal(restored["missing_mask80_test"],window["missing_mask80_test"])

    def test_outer_changes_never_refit_preprocessing_and_keys(self):
        baseline=self.prepare()
        changed=self.frame.copy()
        changed.loc[changed.j>=self.split["v_start_j"],FEATURES]=np.nan
        changed.loc[changed.j>=self.split["v_start_j"],"y_raw"]*=-1
        other=self.prepare(changed)
        self.assertEqual(baseline["preprocessing_hash"],other["preprocessing_hash"])
        self.assertEqual(baseline["frozen_samples"],other["frozen_samples"])
        np.testing.assert_array_equal(baseline["x_train"],other["x_train"])
        self.assertTrue(other["missing_mask_test"].all())

    def test_fixed_missing_label_stops_instead_of_filling_or_replacing(self):
        changed=self.frame.copy()
        key=self.frozen["valid"][0]
        changed.loc[changed.symbol.eq(key["symbol"])&changed.date.eq(key["date"]),"target_valid"]=False
        with self.assertRaisesRegex(ValueError,"mature-label"):
            self.prepare(changed)

    def test_frozen_key_tamper_rejected(self):
        frozen=copy.deepcopy(self.frozen);frozen["train"][0]["date"]="2000-01-01"
        with self.assertRaisesRegex(ValueError,"checksum"):
            self.prepare(frozen=frozen)


if __name__ == "__main__":
    unittest.main()
