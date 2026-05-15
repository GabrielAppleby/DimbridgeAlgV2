import pandas as pd
import numpy as np

def sample_binary_predicates(num_predicates_avg, num_predicates_max, num_binary_avg, num_binary_max):
    predicates = pd.DataFrame(
        np.random.binomial(1, num_binary_avg/num_binary_max, size=(num_binary_max, num_predicates_max))*np.random.binomial(1, num_predicates_avg/num_predicates_max, size=num_predicates_max),
        columns=[f'predicate_{i}' for i in range(num_predicates_max)],
        index=[f'binary_{i}' for i in range(num_binary_max)]
    )
    clauses = {col: predicates[predicates[col]==1].index.tolist() for col in predicates.columns if predicates[col].sum()>0}
    return predicates, clauses

def sample_numeric_attribute(num_numeric_bins_avg, num_numeric_bins_max, size_numeric_bins_max, label):
    numeric_segments = pd.Series(np.random.binomial(1, num_numeric_bins_avg/num_numeric_bins_max, size=num_numeric_bins_max))
    numeric_segment_sizes = np.random.randint(1, size_numeric_bins_max, size=num_numeric_bins_max)
    numeric_segments_ids = numeric_segments[numeric_segments==1].index.values
    numeric_segment_sizes_ranges = numeric_segment_sizes[numeric_segments==1]//2
    numeric_intervals_df = pd.DataFrame(list(zip(numeric_segments_ids-numeric_segment_sizes_ranges, numeric_segments_ids+numeric_segment_sizes_ranges)), columns=['start', 'end'])
    
    numeric_intervals_df[['start','end']] = np.clip(numeric_intervals_df[['start','end']], a_min=0, a_max=num_numeric_bins_max-1)
    merge_id = ((numeric_intervals_df['start']-1) > numeric_intervals_df['end'].shift().cummax()).cumsum()
    numeric_intervals_df_merged = numeric_intervals_df.groupby(merge_id).agg({'start': 'min', 'end': 'max'})
    numeric_intervals = numeric_intervals_df_merged.to_records(index=False).tolist()
    
    interval_ids = np.arange(num_numeric_bins_max)
    interval_values = (
        (interval_ids[:,None]>=numeric_intervals_df_merged['start'].values[None]) &
        (interval_ids[:,None]<=numeric_intervals_df_merged['end'].values[None])
    ).any(axis=1).astype(int)
    predicate = pd.Series(interval_values, index=[f'{label}_{i}' for i in interval_ids])
    return predicate, numeric_intervals

def sample_numeric_predicate(num_numeric_avg, num_numeric_max, num_numeric_bins_avg, num_numeric_bins_max, size_numeric_bins_max, label):
    predicates_list = []
    all_intervals = {}
    for i,include in enumerate(np.random.binomial(1, num_numeric_avg/num_numeric_max, size=num_numeric_max)):
        if include==1:
            predicate, intervals = sample_numeric_attribute(num_numeric_bins_avg, num_numeric_bins_max, size_numeric_bins_max, f'{label}_{i}')
            if len(intervals)>0:
                all_intervals[f'{label}_{i}'] = intervals
        else:
            predicate = pd.Series(
                np.zeros(num_numeric_bins_max).astype(int),
                index=[f'{label}_{i}_{j}' for j in range(num_numeric_bins_max)]
            )
        predicates_list.append(predicate)
    return pd.concat(predicates_list), all_intervals

def sample_numeric_predicates(num_predicates_avg, num_predicates_max, num_numeric_avg, num_numeric_max, num_numeric_bins_avg, num_numeric_bins_max, size_numeric_bins_max):
    all_clauses = {}
    predicates_dict = {}
    for k,include in enumerate(np.random.binomial(1, num_predicates_avg/num_predicates_max, size=num_predicates_max)):
        name = f'predicate_{k}'
        if include==1:
            predicate, clauses = sample_numeric_predicate(num_numeric_avg, num_numeric_max, num_numeric_bins_avg, num_numeric_bins_max, size_numeric_bins_max, 'numeric')
            if len(clauses)>0:
                all_clauses[name] = clauses
            predicates_dict[name] = predicate
        else:
            predicates_dict[name] = pd.Series(np.zeros(num_numeric_max*num_numeric_bins_max).astype(int))
            predicates_dict[name].index = [a for b in [[f'numeric_{i}_{j}' for j in range(num_numeric_bins_max)] for i in range(num_numeric_max)] for a in b]
    return pd.DataFrame(predicates_dict), all_clauses

def sample_numeric_data(num_numeric_avg, num_numeric_max, num, rate, clauses):
    all_cols = [f'numeric_{i}' for i in range(num_numeric_max)]
    if len(clauses)>0:
        data = pd.DataFrame({k:np.random.choice([a for b in [range(vi[0],vi[1]+1) for vi in v] for a in b],size=num) for k,v in clauses.items()})
        cols = [col for col in all_cols if col not in data_in.columns]
        data[cols] = np.random.choice(range(num_numeric_bins_max), size=(num,len(cols)))
        return data.loc[:,all_cols]
    else:
        return pd.DataFrame(np.random.choice(range(num_numeric_max), size=(num,num_numeric_max)), columns=all_cols)

def sample_binary_data(num_binary_avg, num_binary_max, num, predicate):
    data = pd.concat([predicate for i in range(num)], axis=1).T
    data.index = range(len(data))
    cols = data_in.columns[data.isnull().any()]
    data[cols] = np.random.binomial(1, p=num_binary_avg/num_binary_max, size=(num, len(cols)))   
    return data
