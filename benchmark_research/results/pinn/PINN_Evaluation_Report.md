# PINN Evaluation Report
This report audits the Physics-Informed Neural Network after mathematical fixes.

## Evaluation Metrics
| Target | MAE | RMSE | WMAPE | R2 |
|---|---|---|---|---|
| Density | 0.00 | 0.00 | 1.84% | 0.9963 |
| Queue Length | 6.46 | 8.62 | 4.59% | 0.9861 |
| Throughput | 2.40 | 3.14 | 62.26% | 0.3416 |
| Average Speed | 0.64 | 0.88 | 14.04% | 0.8643 |
| Fuel Consumption | 3330.93 | 4407.38 | 3.83% | 0.9818 |

## Diagnostic Plots
### Density
![Density Diagnostics](diagnostics/density_diagnostics.png)
![Density Time Series](diagnostics/density_timeseries.png)

### Queue Length
![Queue Length Diagnostics](diagnostics/queue_length_diagnostics.png)
![Queue Length Time Series](diagnostics/queue_length_timeseries.png)

### Throughput
![Throughput Diagnostics](diagnostics/throughput_diagnostics.png)
![Throughput Time Series](diagnostics/throughput_timeseries.png)

### Average Speed
![Average Speed Diagnostics](diagnostics/average_speed_diagnostics.png)
![Average Speed Time Series](diagnostics/average_speed_timeseries.png)

### Fuel Consumption
![Fuel Consumption Diagnostics](diagnostics/fuel_consumption_diagnostics.png)
![Fuel Consumption Time Series](diagnostics/fuel_consumption_timeseries.png)

## Physics Loss Audit
![Loss Decomposition](diagnostics/loss_decomposition.png)
If the Physics Loss declines alongside Data Loss, it means the network is successfully internalizing the traffic flow continuity equation constraint, anchoring the predictions in reality.