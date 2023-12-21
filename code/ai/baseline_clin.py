import os
import json

import numpy as np
import pandas as pd

from scipy.interpolate import UnivariateSpline

from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer

from calc_cutoffs import get_optimal_cutoffs, apply_cutoffs, estimate_proba

import config


# derivation of the PSI score
# Fine et al. 1997, N Engl J Med || doi:10.1056/nejm199701233360402
# relevance of PSI in the context of COVID-19
# Artero et al. 2021, J Gen Intern Med  || doi:10.1007/s11606-021-06626-7
PSI_COLS_EXTRA = ['pat_sex', 'pat_age', 'comorb_charlson']  # some extra features are needed for informed imputation
PSI_COLS_OUT = ['pneumo_psi-sc']
PSI_COLS_IN = PSI_COLS_OUT + PSI_COLS_EXTRA

PSI_CATEGOR = {'pat_sex': {0: 0, 1: 1}}  # 0: 'm', 1: 'f'


def get_clinical_score_psi(s):
    score_psi = s['pneumo_psi-sc']  # dummy
    return score_psi


# Xie et al. 2020, medRXiv || doi:10.1101/2020.03.28.20045997
XIE_BETAS = {'intercept': +4.559,
             'pat_age': +0.047,  # [years]
             'emerg-status_spo2': -0.098,  # [%]
             'blood-t_ldh': +0.003,  # [U/L]
             'blood-t_lymphoc[ln]': -1.094}  # [10^9/L] (log_e)

XIE_COLS_OUT = list(XIE_BETAS.keys())
XIE_COLS_IN = []
for col_xie in XIE_COLS_OUT:
    if col_xie == 'intercept':
        pass
    elif col_xie == 'blood-t_ldh':
        XIE_COLS_IN.append('blood-t_ldh[log10]')
    elif col_xie == 'blood-t_lymphoc[ln]':
        XIE_COLS_IN.append('blood-t_lymphoc[log10]')
    else:
        XIE_COLS_IN.append(col_xie)

XIE_CATEGOR = dict()


def get_clinical_score_xie(s):
    score_xie = 0
    for col, value in s.items():
        score_xie += XIE_BETAS[col] * value

    return score_xie


# Smith et al. 2019, Clin Med J || doi:10.7861/clinmedicine.19-3-260
NEWS2_POINTS = {'emerg-status_temperat': {(30.0, 35.0): 3, (35.1, 36.0): 1, (36.1, 38.0): 0, (38.1, 39.0): 1, (39.1, 45.0): 2},  # [ºC]
                'emerg-status_bp-syst': {(0, 90): 3, (91, 100): 2, (101, 110): 1, (111, 219): 0, (220, 400): 3},  # [mmHg]
                'emerg-status_heart-rate': {(0, 40): 3, (41, 50): 1, (51, 90): 0, (91, 110): 1, (111, 130): 2, (131, 250): 3},  # [min^-1]
                'emerg-status_resp-rate': {(0, 8): 3, (9, 11): 1, (12, 20): 0, (21, 24): 2, (25, 60): 3},  # [min^-1]
                'emerg-status_spo2': {(0, 91): 3, (92, 93): 2, (94, 95): 1, (96, 100): 0},  # [%]
                'emerg-status_extra-o2': {0: 0, 1: 2},
                'symptoms_confus': {0: 0, 1: 3}}

NEWS2_COLS_IN = list(NEWS2_POINTS.keys())
NEWS2_COLS_OUT = NEWS2_COLS_IN.copy()

NEWS2_CATEGOR = {'emerg-status_extra-o2': {0: 0, 1: 1},  # 0: False, 1: True
                 'symptoms_confus': {0: 0, 1: 1}}  # 0: False, 1: True


def get_clinical_score_news2(s):
    score_news2 = 0
    for col in NEWS2_COLS_IN:
        value = s[col]

        if col in NEWS2_CATEGOR.keys():  # categorical
            score = NEWS2_POINTS[col][int(value)]

        else:  # numeric
            if col == 'emerg-status_temperat':
                value = np.round(value, decimals=1)
            else:
                value = np.round(value, decimals=0).astype(int)

            score = np.nan
            for (val_min, val_max), score_val in NEWS2_POINTS[col].items():
                if (value >= val_min) and (value <= val_max):
                    score = score_val
                    break
            if np.isnan(score):
                raise RuntimeError  # detect out-of-range values

        score_news2 += score

    return score_news2


# Goodacre et al. 2021, PLoS One || doi:10.1371/journal.pone.0245840
PRIEST_POINTS_EXTRA = {'pat_age': {(0, 49): 0, (50, 65): 2, (66, 80): 3, (81, 150): 4},  # [years]
                       'pat_sex': {0: 1, 1: 0},  # 0: 'm', 1: 'f'
                       'pat_phys-status': {'normal': 0, 'light': 1, 'self-care': 2, 'lim_self-care': 3, 'no_self-care': 4}}
PRIEST_POINTS = NEWS2_POINTS.copy()
PRIEST_POINTS.update(PRIEST_POINTS_EXTRA)

PRIEST_COLS_EXTRA_OUT = list(PRIEST_POINTS_EXTRA.keys())
PRIEST_COLS_OUT = NEWS2_COLS_OUT + PRIEST_COLS_EXTRA_OUT
PRIEST_COLS_EXTRA_IN = ['pat_obesity', 'pat_alcohol', 'pat_tobacco', 'pat_nursing',
                        'comorb_charlson']  # proxy info to estimate physical status (not collected explicitly here)
PRIEST_COLS_IN = NEWS2_COLS_IN + PRIEST_COLS_EXTRA_IN + [col for col in PRIEST_COLS_EXTRA_OUT if col != 'pat_phys-status']

PRIEST_CATEGOR_EXTRA = {'pat_sex': {0: 0, 1: 1},  # 0: 'm', 1: 'f'
                        'pat_phys-status': {'normal': 0, 'light': 1, 'self-care': 2, 'lim_self-care': 3, 'no_self-care': 4}}
PRIEST_CATEGOR = NEWS2_CATEGOR.copy()
PRIEST_CATEGOR.update(PRIEST_CATEGOR_EXTRA)


# custom proxy for physical status
# thresholds for categorical Charlson comorbidity index taken from
# Neuhaus et al. 2013, Clin Orthop Relat Res || doi:10.1007/s11999-012-2705-9
def estimate_physical_status_priest(s):
    comorb_charlson_numer = int(s['comorb_charlson'])
    if comorb_charlson_numer < 0:
        raise ValueError
    elif comorb_charlson_numer == 0:
        comorb_charlson_categ = 'no'
    elif 1 <= comorb_charlson_numer <= 2:
        comorb_charlson_categ = 'mild'
    elif 3 <= comorb_charlson_numer <= 4:
        comorb_charlson_categ = 'moderate'
    else:
        comorb_charlson_categ = 'severe'

    pat_obesity = (not np.isnan(s['pat_obesity'])) and (int(s['pat_obesity']) == 1)  # binary
    pat_alcohol = (not np.isnan(s['pat_alcohol'])) and (int(s['pat_alcohol']) == 2)  # 0: no, 1: ex, 2: yes
    pat_tobacco = (not np.isnan(s['pat_tobacco'])) and (int(s['pat_tobacco']) == 2)  # 0: no, 1: ex, 2: yes
    pat_unfit = pat_obesity or pat_alcohol or pat_tobacco

    pat_nursing = (not np.isnan(s['pat_nursing'])) and (int(s['pat_nursing']) == 1)  # binary

    if pat_nursing:
        if comorb_charlson_categ == 'severe':
            pat_phys_status = 'no_self-care'
        else:  # 'no', 'mild', 'moderate'
            pat_phys_status = 'lim_self-care'
    else:
        if pat_unfit:
            pat_phys_status = 'self-care'
        else:
            if comorb_charlson_categ == 'severe':
                pat_phys_status = 'self-care'
            elif comorb_charlson_categ == 'moderate':
                pat_phys_status = 'light'
            else:  # 'no', 'mild'
                pat_phys_status = 'normal'

    return pat_phys_status


def get_clinical_score_priest(s):
    score_priest = get_clinical_score_news2(s)

    for col in PRIEST_COLS_EXTRA_OUT:
        value = s[col]

        if col in PRIEST_CATEGOR_EXTRA.keys():  # categorical
            if col == 'pat_phys-status':
                score = PRIEST_POINTS_EXTRA[col][value]
            else:
                score = PRIEST_POINTS_EXTRA[col][int(value)]

        else:  # numeric
            value = np.round(value, decimals=0).astype(int)

            score = np.nan
            for (val_min, val_max), score_val in PRIEST_POINTS_EXTRA[col].items():
                if (value >= val_min) and (value <= val_max):
                    score = score_val
                    break
            if np.isnan(score):
                raise RuntimeError  # detect out-of-range values

        score_priest += score

    return score_priest


# Gupta et al. 2021, Lancet Respir Med || doi:10.1016/S2213-2600(20)30559-2
ISARIC4C_FILE = './auxil/ISARIC_4C_deterioration_look_up_tables.xlsx'
ISARIC4C_SHEET = 'Predictors'

ISARIC4C_CORRESP = {'Nosocomial infection': 'nosocomial',
                    'Sex': 'pat_sex',
                    'Age (years)': 'pat_age',  # [years]
                    'Respiratory rate (per min)': 'emerg-status_resp-rate',  # [min^-1]
                    'SpO2 (%)': 'emerg-status_spo2',  # [%]
                    'Room air or oxygen': 'emerg-status_extra-o2',
                    'Glasgow coma scale': 'symptoms_confus',
                    'Radiographic infiltrates': 'emerg-pulmo_infiltr-xr',
                    'Urea (mmol/L)': 'blood-t_urea',  # [mmol/L]
                    'C-reactive protein (mg/L)': 'blood-t_crp',  # [mg/L]
                    'Lymphocytes (x10^9/L)': 'blood-t_lymphoc'}  # [10^9 U/L]

ISARIC4C_POINTS_CATEGOR = {'nosocomial': {0: 0, 1: 39},  # 0: False, 1: True
                           'pat_sex': {0: 35, 1: 0},  # 0: 'm', 1: 'f'
                           'emerg-pulmo_infiltr-xr': {0: 0, 1: 47},  # 0: False, 1: True
                           'emerg-status_extra-o2': {0: 0, 1: 108},  # 0: False, 1: True
                           'symptoms_confus': {0: 0, 1: 87}}  # 0: False, 1: True

ISARIC4C_CATEGOR = {'nosocomial': {0: 0, 1: 1},  # 0: False, 1: True
                    'pat_sex': {0: 0, 1: 1},  # 0: 'm', 1: 'f'
                    'emerg-pulmo_infiltr-xr': {0: 0, 1: 1},  # 0: False, 1: True
                    'emerg-status_extra-o2': {0: 0, 1: 1},  # 0: False, 1: True
                    'symptoms_confus': {0: 0, 1: 1}}  # 0: False, 1: True
ISARIC4C_COLS_CATEGOR = list(ISARIC4C_CATEGOR.keys())


def prepare_isaric4c():
    with open(ISARIC4C_FILE, 'rb') as f_lookup:
        df_lookup = pd.read_excel(f_lookup, sheet_name=ISARIC4C_SHEET)

    l_points = ISARIC4C_POINTS_CATEGOR.copy()
    l_predictors = df_lookup['predictor'].unique()
    for pred_temp in l_predictors:
        if ISARIC4C_CORRESP[pred_temp] in l_points:
            continue

        idx_sel = (df_lookup['predictor'] == pred_temp)
        x_temp = df_lookup.loc[idx_sel, 'value'].to_numpy().astype(float)
        y_temp = df_lookup.loc[idx_sel, 'score'].to_numpy().astype(float)

        f_temp = UnivariateSpline(x_temp, y_temp, ext='extrapolate')
        l_points[ISARIC4C_CORRESP[pred_temp]] = f_temp

    return l_points


ISARIC4C_POINTS = prepare_isaric4c()

ISARIC4C_COLS_OUT = list(ISARIC4C_POINTS.keys())
ISARIC4C_COLS_IN = []
for col_isaric4c in ISARIC4C_COLS_OUT:
    if col_isaric4c == 'nosocomial':
        pass
    elif col_isaric4c == 'blood-t_urea':
        ISARIC4C_COLS_IN.append('blood-t_urea[log10]')
    elif col_isaric4c == 'blood-t_crp':
        ISARIC4C_COLS_IN.append('blood-t_crp[log10]')
    elif col_isaric4c == 'blood-t_lymphoc':
        ISARIC4C_COLS_IN.append('blood-t_lymphoc[log10]')
    else:
        ISARIC4C_COLS_IN.append(col_isaric4c)


def get_clinical_score_isaric4c(s):
    score_isaric4c = 0
    for col, value in s.items():
        if col in ISARIC4C_COLS_CATEGOR:
            score = ISARIC4C_POINTS[col][int(value)]
        else:
            f_points = ISARIC4C_POINTS[col]
            score = f_points(value)
            score = int(np.round(score))
        score_isaric4c += score

    return score_isaric4c


# impute missing values
SCALER_QUANTILES = (25.0, 75.0)

IMPUTER_NEIGHBORS = 9
IMPUTER_WEIGHTS = 'distance'
# IMPUTER_NEIGHBORS = 5
# IMPUTER_WEIGHTS = 'uniform'


def impute_data(data_in, replace_categ):
    # replace categorical by numeric wherever required
    data_temp = data_in.replace(replace_categ)

    # scale, impute, undo scaling
    scaler = RobustScaler(quantile_range=SCALER_QUANTILES)
    imputer = KNNImputer(n_neighbors=IMPUTER_NEIGHBORS, weights=IMPUTER_WEIGHTS)

    data_cols = data_temp.columns
    data_temp = data_temp.to_numpy()
    data_temp = scaler.fit_transform(data_temp)
    data_temp = imputer.fit_transform(data_temp)
    data_temp = scaler.inverse_transform(data_temp)

    data_out = pd.DataFrame(data=data_temp, columns=data_cols)

    # inverse the replacement of categorical
    replace_inverse = dict()
    for col in data_out.columns:
        if col in replace_categ:
            data_out[col] = data_out[col].round().astype(int)
            replace_inverse[col] = dict((val, key) for key, val in replace_categ[col].items())
    data_out = data_out.replace(replace_inverse)

    return data_out


# load data
def load_data(filename):
    df_data = pd.read_csv(filename, header=0, sep=config.DELIMITER)
    if len(df_data.columns) == 1:
        raise SystemExit(config.MSSG_ERROR_DATA)
    return df_data


def save_results(method, y_true, y_pred, score_pred, proba_pred, cutoffs):
    results = {'y_true': y_true.tolist(),
               'y_pred': y_pred.tolist(), 'score_pred': score_pred.tolist(), 'proba_pred': proba_pred.tolist(),
               'cutoffs': cutoffs.tolist()}

    # determine filename
    path = config.PATH_RESULTS_BASELINE
    filename = config.FILE_RESULTS_BASELINE.format('baseline-clin_{}'.format(method))

    # store to disk via json
    f_results = os.path.join(path, filename)
    with open(f_results, 'w') as f_json_results:
        json.dump(results, f_json_results)

    return f_results


# whether to use class priors as weights in the search of optimal cut-offs
WEIGH_CLASSES = False
# WEIGH_CLASSES = True


if __name__ == '__main__':
    # import data file(s)
    data_main = load_data(filename=config.FILE_DATA_IN)
    df_X = data_main.drop(columns=[config.VAR_CLASSIF])
    df_y = data_main[config.VAR_CLASSIF]

    data_suppl = load_data(filename=config.FILE_DATA_SUPPL)
    col_suppl = 'emerg-status_extra-o2'
    idx_loc = df_X.columns.get_loc('emerg-status_spo2') + 1
    df_X.insert(idx_loc, col_suppl, data_suppl[col_suppl])

    # compute class priors
    priors = np.ones(shape=(config.NUM_CLASSES,))
    for idx_l, label in enumerate(config.LABELS):
        priors[idx_l] = (df_y == label).sum()
    priors /= np.sum(priors)

    # compute method by method
    l_methods = ['PSI', 'Xie', 'NEWS2', 'PRIEST', 'ISARIC4C']
    l_cols_in = [PSI_COLS_IN, XIE_COLS_IN, NEWS2_COLS_IN, PRIEST_COLS_IN, ISARIC4C_COLS_IN]
    l_cols_out = [PSI_COLS_OUT, XIE_COLS_OUT, NEWS2_COLS_OUT, PRIEST_COLS_OUT, ISARIC4C_COLS_OUT]
    l_repl_cat = [PSI_CATEGOR, XIE_CATEGOR, NEWS2_CATEGOR, PRIEST_CATEGOR, ISARIC4C_CATEGOR]
    for method, cols_in, cols_out, repl_cat in zip(l_methods, l_cols_in, l_cols_out, l_repl_cat):
        df_X_sel = df_X.copy()
        y_true = df_y.to_numpy()

        # retain only the strictly necessary information
        df_X_sel = df_X_sel[cols_in]

        # apply required transforms on inputs, including auxiliary variables
        if method in ['PSI', 'Xie', 'NEWS2']:
            pass
        elif method == 'PRIEST':
            df_X_sel['pat_phys-status'] = df_X_sel.apply(estimate_physical_status_priest, axis='columns')
            df_X_sel = df_X_sel.drop(columns=PRIEST_COLS_EXTRA_IN)
        elif method == 'ISARIC4C':
            replace_infiltr = {'Missing': np.nan, 'No': 0,
                               'Unilobar': 1, 'Multilob unilat': 1, 'Bilateral': 1}
            df_X_sel = df_X_sel.replace({'emerg-pulmo_infiltr-xr': replace_infiltr})
        else:
            raise NotImplementedError

        # impute missing values
        df_X_sel = impute_data(df_X_sel, replace_categ=repl_cat)

        # apply required transforms on inputs, including management of units and natural/logarithmic scales
        if method in ['PSI', 'NEWS2', 'PRIEST']:
            pass
        elif method == 'Xie':
            df_X_sel['intercept'] = 1.0  # add intercept
            df_X_sel['blood-t_ldh'] = np.power(10.0, df_X_sel['blood-t_ldh[log10]'])
            df_X_sel['blood-t_lymphoc'] = np.power(10.0, df_X_sel['blood-t_lymphoc[log10]'])  # undo log_10 logarithmic transform
            df_X_sel['blood-t_lymphoc'] = 1.0e-3 * df_X_sel['blood-t_lymphoc']  # change units: from [10^6 U/L] to [10^9 U/L]
            df_X_sel['blood-t_lymphoc[ln]'] = np.log(df_X_sel['blood-t_lymphoc'])  # change logarithm base to log_e
        elif method == 'ISARIC4C':
            df_X_sel['nosocomial'] = 0  # all our patients were non-nosocomial by inclusion criteria
            df_X_sel['blood-t_urea'] = np.power(10.0, df_X_sel['blood-t_urea[log10]'])  # undo log_10 logarithmic transform
            df_X_sel['blood-t_urea'] = (1 / 6.006) * df_X_sel['blood-t_urea']  # change units: from [mg/dL] to [mmol/L]
            df_X_sel['blood-t_crp'] = np.power(10.0, df_X_sel['blood-t_crp[log10]'])  # undo log_10 logarithmic transform
            df_X_sel['blood-t_lymphoc'] = np.power(10.0, df_X_sel['blood-t_lymphoc[log10]'])  # undo log_10 logarithmic transform
            df_X_sel['blood-t_lymphoc'] = 1.0e-3 * df_X_sel['blood-t_lymphoc']  # change units: from [10^6 U/L] to [10^9 U/L]
        else:
            raise NotImplementedError

        # retain only the strictly necessary information
        df_X_sel = df_X_sel[cols_out]

        # compute clinical scoring
        if method == 'PSI':
            def get_clinical_score(row): return get_clinical_score_psi(row)
        elif method == 'Xie':
            def get_clinical_score(row): return get_clinical_score_xie(row)
        elif method == 'NEWS2':
            def get_clinical_score(row): return get_clinical_score_news2(row)
        elif method == 'PRIEST':
            def get_clinical_score(row): return get_clinical_score_priest(row)
        elif method == 'ISARIC4C':
            def get_clinical_score(row): return get_clinical_score_isaric4c(row)
        else:
            raise NotImplementedError

        score_pred = df_X_sel.apply(get_clinical_score, axis='columns')
        score_pred = score_pred.to_numpy()

        # weigh by class priors, if required
        weights = np.copy(priors) if WEIGH_CLASSES else None

        # determine optimal cut-offs for the score
        cutoffs, l_distrib = get_optimal_cutoffs(x=score_pred, y=y_true, weights=weights)

        # apply those cut-offs in prediction
        y_pred = apply_cutoffs(score_pred, cutoffs)

        # estimate class probabilities
        proba_pred = estimate_proba(score_pred, l_distrib, priors)

        # save to file
        save_results(method, y_true, y_pred, score_pred, proba_pred, cutoffs)

    print('Done!')
