#For each community area + crime type, calculate average from training data - Seasonal adjustment: average for same month (e.g., all Januaries) - Predictions generated for test set (same scope as KNN) - Accuracy metrics calculated: MAE, RMSE - Runtime: <10 seconds for full test
#set - Results stored for comparison
#create HistoricalAverage class
#2. Aggregate training data: group by community area, month, crime type 3. Cal- culate mean crime count per group 4. Implement predict method: lookup average for given area/month 5. Generate predictions for test set 6. Calculate MAE, RMSE 7. Compare to KNN results 8. Document in code comments

class HistoricalAverage:
    gr
