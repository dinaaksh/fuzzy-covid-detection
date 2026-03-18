import numpy as np
import pandas as pd
import skfuzzy as fuzz
import skfuzzy.control as ctrl
from tqdm import tqdm


class FuzzyOntologyEngine:
    def __init__(self):
        self._build_universes()
        self._build_systems()

    def _build_universes(self):
        #0–1=mild, 2–4=moderate, 5+=heavy
        self.symptom_count_in = ctrl.Antecedent(np.linspace(0, 9, 91), 'symptom_count')
        self.symptom_count_in['low']    = fuzz.trapmf(self.symptom_count_in.universe, [0, 0, 1, 2])
        self.symptom_count_in['medium'] = fuzz.trimf(self.symptom_count_in.universe,  [1, 3, 5])
        self.symptom_count_in['high']   = fuzz.trapmf(self.symptom_count_in.universe, [4, 6, 9, 9])

        self.symptom_burden = ctrl.Consequent(np.linspace(0, 1, 101), 'symptom_burden',
                                              defuzzify_method='centroid')
        self.symptom_burden['low']    = fuzz.trapmf(self.symptom_burden.universe, [0.0, 0.0, 0.25, 0.40])
        self.symptom_burden['medium'] = fuzz.trimf(self.symptom_burden.universe,  [0.25, 0.50, 0.75])
        self.symptom_burden['high']   = fuzz.trapmf(self.symptom_burden.universe, [0.60, 0.80, 1.0, 1.0])

        #ratios: loss_of_smell 22x, loss_of_taste 18x, fever 8.6x, muscle_sore 5.2x, cough 5.1x
        self.covid_sig_in = ctrl.Antecedent(np.linspace(0, 5, 51), 'covid_signature')
        self.covid_sig_in['low']    = fuzz.trapmf(self.covid_sig_in.universe, [0, 0, 0.5, 1.5])
        self.covid_sig_in['medium'] = fuzz.trimf(self.covid_sig_in.universe,  [1.0, 2.0, 3.0])
        self.covid_sig_in['high']   = fuzz.trapmf(self.covid_sig_in.universe, [2.5, 3.5, 5.0, 5.0])

        self.covid_signature = ctrl.Consequent(np.linspace(0, 1, 101), 'covid_signature_score',
                                               defuzzify_method='centroid')
        self.covid_signature['low']    = fuzz.trapmf(self.covid_signature.universe, [0.0, 0.0, 0.20, 0.40])
        self.covid_signature['medium'] = fuzz.trimf(self.covid_signature.universe,  [0.25, 0.50, 0.75])
        self.covid_signature['high']   = fuzz.trapmf(self.covid_signature.universe, [0.60, 0.80, 1.0, 1.0])

        self.temp = ctrl.Antecedent(np.linspace(35, 42, 71), 'temperature')
        self.temp['normal']   = fuzz.trapmf(self.temp.universe, [35.0, 35.0, 37.2, 37.5])
        self.temp['elevated'] = fuzz.trimf(self.temp.universe,  [37.2, 37.8, 38.5])
        self.temp['high']     = fuzz.trapmf(self.temp.universe, [38.0, 39.0, 42.0, 42.0])

        self.pulse = ctrl.Antecedent(np.linspace(40, 200, 161), 'pulse')
        self.pulse['normal']   = fuzz.trapmf(self.pulse.universe, [40, 40,  90, 100])
        self.pulse['elevated'] = fuzz.trimf(self.pulse.universe,  [90, 110, 125])
        self.pulse['high']     = fuzz.trapmf(self.pulse.universe, [115, 130, 200, 200])

        self.sats = ctrl.Antecedent(np.linspace(80, 100, 21), 'sats')
        self.sats['low']    = fuzz.trapmf(self.sats.universe, [80, 80, 92, 94])
        self.sats['normal'] = fuzz.trapmf(self.sats.universe, [93, 95, 100, 100])

        self.vitals_risk = ctrl.Consequent(np.linspace(0, 1, 101), 'vitals_risk',
                                           defuzzify_method='centroid')
        self.vitals_risk['low']    = fuzz.trapmf(self.vitals_risk.universe, [0.0, 0.0, 0.25, 0.40])
        self.vitals_risk['medium'] = fuzz.trimf(self.vitals_risk.universe,  [0.25, 0.50, 0.75])
        self.vitals_risk['high']   = fuzz.trapmf(self.vitals_risk.universe, [0.60, 0.80, 1.0, 1.0])


        for name in ['in_burden', 'in_signature', 'in_vitals']:
            inp = ctrl.Antecedent(np.linspace(0, 1, 101), name)
            inp['low']    = fuzz.trapmf(inp.universe, [0.0, 0.0, 0.25, 0.45])
            inp['medium'] = fuzz.trimf(inp.universe,  [0.25, 0.50, 0.75])
            inp['high']   = fuzz.trapmf(inp.universe, [0.55, 0.75, 1.0, 1.0])
            setattr(self, name, inp)

        self.final_risk = ctrl.Consequent(np.linspace(0, 1, 101), 'final_risk',
                                          defuzzify_method='centroid')
        self.final_risk['low']    = fuzz.trapmf(self.final_risk.universe, [0.0, 0.0, 0.25, 0.45])
        self.final_risk['medium'] = fuzz.trimf(self.final_risk.universe,  [0.30, 0.50, 0.70])
        self.final_risk['high']   = fuzz.trapmf(self.final_risk.universe, [0.55, 0.75, 1.0, 1.0])

    def _build_systems(self):

        sc = self.symptom_count_in
        sb = self.symptom_burden
        burden_rules = [
            ctrl.Rule(sc['high'],   sb['high']),
            ctrl.Rule(sc['medium'], sb['medium']),
            ctrl.Rule(sc['low'],    sb['low']),
        ]
        self.burden_sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem(burden_rules))

        cs = self.covid_sig_in
        cso = self.covid_signature
        signature_rules = [
            ctrl.Rule(cs['high'],   cso['high']),
            ctrl.Rule(cs['medium'], cso['medium']),
            ctrl.Rule(cs['low'],    cso['low']),
        ]
        self.signature_sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem(signature_rules))

        t, p, s, vr = self.temp, self.pulse, self.sats, self.vitals_risk
        vitals_rules = [
            ctrl.Rule(s['low'],vr['high']), 
            ctrl.Rule(t['high']  | p['high'],vr['high']), 
            ctrl.Rule(t['elevated'] & p['elevated'],vr['medium']), 
            ctrl.Rule(t['elevated'] | p['elevated'],vr['medium']),  
            ctrl.Rule(t['normal'] & p['normal'] & s['normal'], vr['low']),
        ]
        self.vitals_sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem(vitals_rules))

        ib, isig, iv, fr = self.in_burden, self.in_signature, self.in_vitals, self.final_risk
        master_rules = [
            ctrl.Rule(isig['high'],                      fr['high']),
            ctrl.Rule(isig['medium'] & ib['high'],       fr['high']),
            ctrl.Rule(isig['medium'] & iv['high'],       fr['high']),
            ctrl.Rule(isig['medium'] & ib['medium'],     fr['medium']),
            ctrl.Rule(isig['medium'] & iv['medium'],     fr['medium']),
            ctrl.Rule(isig['low']   & ib['high'],        fr['medium']),
            ctrl.Rule(isig['low']   & iv['high'],        fr['medium']),
            ctrl.Rule(isig['low']   & ib['medium'],      fr['low']),
            ctrl.Rule(isig['low']   & ib['low'],         fr['low']),
        ]
        self.master_sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem(master_rules))


    def _safe_compute(self, sim, inputs: dict, output_key: str) -> float:
        try:
            for k, v in inputs.items():
                sim.input[k] = float(v)
            sim.compute()
            return float(sim.output[output_key])
        except Exception:
            return 0.0 

    def score_row(self, row: dict) -> dict:

        symptom_count = float(row.get('symptom_count', 0))
        burden_score = self._safe_compute(
            self.burden_sim,
            {'symptom_count': np.clip(symptom_count, 0, 9)},
            'symptom_burden'
        )

        covid_sig = sum([
            float(row.get('loss_of_smell', 0)),
            float(row.get('loss_of_taste', 0)),
            float(row.get('fever', 0)),
            float(row.get('cough', 0)),
            float(row.get('muscle_sore', 0)),
        ])
        signature_score = self._safe_compute(
            self.signature_sim,
            {'covid_signature': np.clip(covid_sig, 0, 5)},
            'covid_signature_score'
        )

        vitals_measured = (
            float(row.get('temperature_measured', 0)) > 0 or
            float(row.get('pulse_measured', 0)) > 0 or
            float(row.get('sats_measured', 0)) > 0
        )

        if vitals_measured:
            temp_val  = float(row.get('temperature', 0)) if float(row.get('temperature_measured', 0)) else 37.0
            pulse_val = float(row.get('pulse', 0))       if float(row.get('pulse_measured', 0))       else 75.0
            sats_val  = float(row.get('sats', 0))        if float(row.get('sats_measured', 0))         else 98.0

            vitals_score = self._safe_compute(
                self.vitals_sim,
                {
                    'temperature': np.clip(temp_val,  35, 42),
                    'pulse':       np.clip(pulse_val, 40, 200),
                    'sats':        np.clip(sats_val,  80, 100),
                },
                'vitals_risk'
            )
        else:
            vitals_score = 0.0

        final_score = self._safe_compute(
            self.master_sim,
            {
                'in_burden':    burden_score,
                'in_signature': signature_score,
                'in_vitals':    vitals_score,
            },
            'final_risk'
        )

        return {
            'fis_symptom_burden':   burden_score,  
            'fis_covid_signature':  signature_score,  
            'fis_vitals_risk':      vitals_score,    
            'fis_covid_score':      final_score,      
        }

    def score_row_with_memberships(self, row: dict) -> dict:

        scores = self.score_row(row)
        crisp_output = scores['fis_covid_score']
        universe     = self.final_risk.universe

        memberships = {
            'low':    float(fuzz.interp_membership(
                          universe, self.final_risk['low'].mf,    crisp_output)),
            'medium': float(fuzz.interp_membership(
                          universe, self.final_risk['medium'].mf, crisp_output)),
            'high':   float(fuzz.interp_membership(
                          universe, self.final_risk['high'].mf,   crisp_output)),
        }
        driver = max(
            [
                ('covid_signature', scores['fis_covid_signature']),
                ('symptom_burden',  scores['fis_symptom_burden']),
                ('vitals_risk',     scores['fis_vitals_risk']),
            ],
            key=lambda x: x[1]
        )[0]

        return {**scores, 'memberships': memberships, 'primary_driver': driver}

    def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        print("   > Running Hierarchical Fuzzy Inference System (HFIS)...")
        records = df.to_dict('records')
        results = [self.score_row(row) for row in tqdm(records, desc="HFIS Scoring")]
        return pd.DataFrame(results, index=df.index)