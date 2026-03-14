import numpy as np
import pandas as pd
import skfuzzy as fuzz
import skfuzzy.control as ctrl
from tqdm import tqdm

class FuzzyOntologyEngine:
    def __init__(self):
        self._build_universes()
        self._build_membership_functions()
        self._build_systems()

    def _build_universes(self):
        binary_inputs = [
            'smell_loss', 'taste_loss', 'fever', 'cough', 'sob',
            'fatigue', 'runny_nose', 'sore_throat', 'headache',
            'diarrhea', 'muscle_sore', 'exposure'
        ]
        self._antecedents = {}
        for name in binary_inputs:
            u = ctrl.Antecedent(np.linspace(0, 1, 11), name)
            u['absent']  = fuzz.trimf(u.universe, [0.0, 0.0, 0.5])
            u['present'] = fuzz.trimf(u.universe, [0.5, 1.0, 1.0])
            self._antecedents[name] = u

        self.temp_u = ctrl.Antecedent(np.linspace(35, 42, 71), 'temperature')
        self.sats_u = ctrl.Antecedent(np.linspace(85, 100, 151), 'sats')
        self.rr_u   = ctrl.Antecedent(np.linspace(8, 40, 321), 'rr')

        self.covid_risk_u    = ctrl.Consequent(np.linspace(0, 1, 101), 'covid_risk',    defuzzify_method='centroid')
        self.flu_risk_u      = ctrl.Consequent(np.linspace(0, 1, 101), 'flu_risk',      defuzzify_method='centroid')
        self.resp_alarm_u    = ctrl.Consequent(np.linspace(0, 1, 101), 'resp_alarm',    defuzzify_method='centroid')
        self.exposure_out_u  = ctrl.Consequent(np.linspace(0, 1, 101), 'exposure_out',  defuzzify_method='centroid')

    def _build_membership_functions(self):
        t = self.temp_u
        t['normal']    = fuzz.trapmf(t.universe, [35, 35, 37.2, 37.5])
        t['low_grade'] = fuzz.trimf(t.universe,  [37.2, 37.8, 38.5])
        t['high']      = fuzz.trapmf(t.universe, [38.0, 38.5, 42, 42])

        s = self.sats_u
        s['normal']     = fuzz.trapmf(s.universe, [95, 97, 100, 100])
        s['concerning'] = fuzz.trimf(s.universe,  [92, 94,  96])
        s['critical']   = fuzz.trapmf(s.universe, [85, 85,  92,  94])

        r = self.rr_u
        r['normal']   = fuzz.trapmf(r.universe, [8,  8,  16, 20])
        r['elevated'] = fuzz.trapmf(r.universe, [18, 22, 40, 40])

        for out_u in [self.covid_risk_u, self.flu_risk_u, self.resp_alarm_u, self.exposure_out_u]:
            out_u['low']    = fuzz.trapmf(out_u.universe, [0.0, 0.0, 0.2, 0.35])
            out_u['medium'] = fuzz.trimf(out_u.universe,  [0.25, 0.5, 0.75])
            out_u['high']   = fuzz.trapmf(out_u.universe, [0.65, 0.8, 1.0, 1.0])

    def _a(self, name):
        return self._antecedents[name]

    def _build_systems(self):
        sl, tl, fv, cg, sb, ft, rn, st, hd, dr, ms, ex = (
            self._a('smell_loss'), self._a('taste_loss'), self._a('fever'), self._a('cough'), 
            self._a('sob'), self._a('fatigue'), self._a('runny_nose'), self._a('sore_throat'), 
            self._a('headache'), self._a('diarrhea'), self._a('muscle_sore'), self._a('exposure')
        )
        tmp, sp, rr = self.temp_u, self.sats_u, self.rr_u
        cr, fr, ra, eo = self.covid_risk_u, self.flu_risk_u, self.resp_alarm_u, self.exposure_out_u

        covid_rules = [
            ctrl.Rule(ft['absent'] & cg['absent'] & hd['absent'] & sb['absent'], cr['medium']),
            
            ctrl.Rule(ft['absent'] & cg['absent'] & hd['present'] & sl['present'], cr['high']),
            ctrl.Rule(ft['absent'] & cg['present'] & sl['present'], cr['high']),
            ctrl.Rule(ft['present'] & sl['present'] & ex['absent'], cr['high']),
            ctrl.Rule(ft['present'] & sl['present'] & ex['present'] & sb['absent'], cr['high']),
            
            ctrl.Rule(ft['absent'] & cg['present'] & sl['absent'] & fv['present'], cr['high']),
            
            ctrl.Rule(ft['present'] & sl['absent'] & fv['present'] & cg['present'], cr['high']),
            
            ctrl.Rule(ft['absent'] & cg['absent'] & hd['absent'] & sb['present'], cr['low']),
            ctrl.Rule(ft['absent'] & cg['absent'] & hd['present'] & sl['absent'], cr['low']),
            ctrl.Rule(ft['absent'] & cg['present'] & sl['absent'] & fv['absent'], cr['low']),
            
            ctrl.Rule(ft['present'] & sl['absent'] & fv['absent'], cr['low']),
            ctrl.Rule(ft['present'] & sl['absent'] & fv['present'] & cg['absent'], cr['low']),
            
            ctrl.Rule(ft['present'] & sl['present'] & ex['present'] & sb['present'], cr['low'])
        ]

        flu_rules = [
            ctrl.Rule(rn['present'] & st['present'] & hd['present'],             fr['high']),
            ctrl.Rule(rn['present'] & st['present'],                             fr['medium']),
            ctrl.Rule(rn['present'] & hd['present'],                             fr['medium']),
            ctrl.Rule(rn['present'] & sl['absent'],                              fr['medium']),
            ctrl.Rule(st['present'] & sl['absent'] & tl['absent'],               fr['medium']),
            ctrl.Rule(sb['present'] & rn['present'],                             fr['medium']),
            ctrl.Rule(ft['present'] & st['present'],                             fr['medium']),
            ctrl.Rule(sl['present'],                                             fr['low']),
            ctrl.Rule(sl['present'] & tl['present'],                             fr['low']),
        ]

        resp_rules = [
            ctrl.Rule(sp['critical'],                                            ra['high']),
            ctrl.Rule(sp['concerning'] & rr['elevated'],                         ra['high']),
            ctrl.Rule(sp['concerning'] & sb['present'],                          ra['medium']),
            ctrl.Rule(rr['elevated'] & sb['present'],                            ra['medium']),
            ctrl.Rule(tmp['high'] & rr['elevated'],                              ra['medium']),
            ctrl.Rule(sp['normal'] & rr['normal'],                               ra['low']),
        ]

        exposure_rules = [
            ctrl.Rule(ex['present'] & fv['present'],                             eo['high']),
            ctrl.Rule(ex['present'] & cg['present'],                             eo['medium']),
            ctrl.Rule(ex['present'],                                             eo['medium']),
            ctrl.Rule(ex['absent'],                                              eo['low']),
        ]

        self.covid_sys    = ctrl.ControlSystem(covid_rules)
        self.flu_sys      = ctrl.ControlSystem(flu_rules)
        self.resp_sys     = ctrl.ControlSystem(resp_rules)
        self.exposure_sys = ctrl.ControlSystem(exposure_rules)
        
        self.covid_sim    = ctrl.ControlSystemSimulation(self.covid_sys)
        self.flu_sim      = ctrl.ControlSystemSimulation(self.flu_sys)
        self.resp_sim     = ctrl.ControlSystemSimulation(self.resp_sys)
        self.exposure_sim = ctrl.ControlSystemSimulation(self.exposure_sys)

    def _safe_simulate(self, sim, inputs: dict, output_key: str) -> float:
        try:
            for k, v in inputs.items(): 
                sim.input[k] = float(v)
            sim.compute()
            return float(sim.output[output_key])
        except Exception: 
            return 0.0

    def score_row(self, row: dict) -> dict:
        def b(col): return float(np.clip(row.get(col, 0), 0, 1))

        temp = float(np.clip(row.get('temperature', 37.0), 35, 42))
        sats = float(np.clip(row.get('sats', 98.0), 85, 100))
        rr   = float(np.clip(row.get('rr',   16.0),  8, 40))

        covid_score = self._safe_simulate(self.covid_sim, {
            'smell_loss': b('loss_of_smell'), 'taste_loss': b('loss_of_taste'), 'fever': b('fever'),
            'sob': b('sob'), 'fatigue': b('fatigue'), 'diarrhea': b('diarrhea'), 'muscle_sore': b('muscle_sore'),
            'runny_nose': b('runny_nose'), 'sore_throat': b('sore_throat'), 'exposure': b('exposure_risk'),
        }, 'covid_risk')

        flu_score = self._safe_simulate(self.flu_sim, {
            'runny_nose': b('runny_nose'), 'sore_throat': b('sore_throat'), 'headache': b('headache'),
            'sob': b('sob'), 'fatigue': b('fatigue'), 'smell_loss': b('loss_of_smell'), 'taste_loss': b('loss_of_taste'),
        }, 'flu_risk')

        resp_score = self._safe_simulate(self.resp_sim, {
            'sats': sats, 'rr': rr, 'sob': b('sob'), 'temperature': temp,
        }, 'resp_alarm')

        exp_score = self._safe_simulate(self.exposure_sim, {
            'exposure': b('exposure_risk'), 'fever': b('fever'), 'cough': b('cough'),
        }, 'exposure_out')

        return {
            'fis_covid_score': covid_score, 'fis_flu_score': flu_score,
            'fis_resp_alarm': resp_score, 'fis_exposure_score': exp_score,
            'fis_covid_vs_flu': covid_score - flu_score,
        }

    def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        print("   > Running Fuzzy Ontology Engine (Calculating 56,000 integrals)...")
        records = df.to_dict('records')
        
        results = [self.score_row(row) for row in tqdm(records, desc="Fuzzy Scoring")]
        
        return pd.DataFrame(results, index=df.index)