# EDA summary

## Cleaning report

- rows_in: 5000
- duplicates_dropped: 0
- out_of_range_set_nan: {'finishing_temp_c': 12}
- missing_target_dropped: 10
- imputed_cells: 416
- rows_out: 4990

## Spec compliance in data (YS)

| Grade | n | mean | std | below min % | above band % |
|---|---|---|---|---|---|
| Fe500 | 1284 | 523.0 | 22.7 | 14.0 | 1.7 |
| Fe500D | 1992 | 538.1 | 22.0 | 2.6 | 3.6 |
| Fe550D | 1221 | 580.5 | 22.9 | 8.2 | 2.1 |
| Fe600 | 493 | 614.8 | 23.5 | 27.0 | 0.6 |

## Top within-grade correlations with YS

- quench_index: +0.62
- finishing_temp_c: -0.58
- surface_to_volume: +0.54
- mill_speed_mps: +0.53
- bar_dia_mm: -0.52
- inv_speed: -0.51
- discharge_temp_c: -0.48
- soaking_temp_c: -0.47
