# Parameter influence ranking (controllable RHF / mill / water-box parameters)

| Rank | Parameter | Group | MAE increase if shuffled (MPa) | dYS per +1 std (MPa) | dYS per unit (MPa) |
|---|---|---|---|---|---|
| 1 | bar_dia_mm | Mill | 8.48 | -9.7 | -1.56 |
| 2 | finishing_temp_c | Mill | 5.03 | -8.3 | -0.46 |
| 3 | wb_flow_m3h | Water box | 2.60 | +4.8 | +0.13 |
| 4 | wb_pressure_bar | Water box | 1.15 | +2.7 | +1.68 |
| 5 | rhf_residence_min | RHF | 0.61 | -2.0 | -0.12 |
| 6 | mill_speed_mps | Mill | 0.46 | +1.4 | +0.46 |
| 7 | discharge_temp_c | RHF | 0.44 | -1.9 | -0.09 |
| 8 | soaking_temp_c | RHF | 0.25 | -1.8 | -0.09 |
| 9 | wb_water_temp_c | Water box | 0.24 | -1.6 | -0.35 |
| 10 | wb_valve_open_pct | Water box | 0.02 | +0.2 | +0.03 |
| 11 | ambient_temp_c | Mill | 0.00 | +0.2 | +0.03 |
| 12 | heating_zone_temp_c | RHF | -0.00 | -0.1 | -0.01 |
| 13 | soaking_zone_temp_c | RHF | -0.00 | -0.2 | -0.01 |
| 14 | billet_size_mm | RHF | -0.00 | +0.0 | +0.00 |
| 15 | preheat_zone_temp_c | RHF | -0.03 | +0.1 | +0.00 |

Chemistry and grade are ranked separately in `permutation_importance.csv`.
