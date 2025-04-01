import os
import joblib
import json
import pandas as pd
import logging
from azureml.core.model import Model


def init():
    global model

        
    model_path = Model.get_model_path('model')
    model = joblib.load(model_path)


def run(raw_data):
    try:
        data = json.loads(raw_data)['data'][0]
        data = pd.DataFrame(data)
        data_df = data_df.drop(columns=[
        " ROA(C) before interest and depreciation before interest",
        " ROA(A) before interest and % after tax",
        " ROA(B) before interest and depreciation after tax",
        " Operating Gross Margin",
        " Realized Sales Gross Margin",
        " Operating Profit Rate",
        " Pre-tax net Interest Rate",
        " After-tax net Interest Rate",
        " Non-industry income and expenditure/revenue",
        " Continuous interest rate (after tax)",
        " Cash flow rate",
        " Net Value Per Share (B)",
        " Net Value Per Share (A)",
        " Net Value Per Share (C)",
        " Persistent EPS in the Last Four Seasons",
        " Operating Profit Per Share (Yuan \u00a5)",
        " Per Share Net profit before tax (Yuan \u00a5)",
        " After-tax Net Profit Growth Rate",
        " Regular Net Profit Growth Rate",
        " Debt ratio %",
        " Net worth/Assets",
        " Borrowing dependency",
        " Contingent liabilities/Net worth",
        " Operating profit/Paid-in capital",
        " Net profit before tax/Paid-in capital",
        " Inventory and accounts receivable/Net value",
        " Total Asset Turnover",
        " Net Worth Turnover Rate (times)",
        " Revenue per person",
        " Working Capital to Total Assets",
        " Current Assets/Total Assets",
        " Current Liability to Assets",
        " Operating Funds to Liability",
        " Current Liabilities/Liability",
        " Working Capital/Equity",
        " Current Liabilities/Equity",
        " Retained Earnings to Total Assets",
        " Total expense/Assets",
        " Working capitcal Turnover Rate",
        " Cash Flow to Sales",
        " Current Liability to Liability",
        " Current Liability to Equity",
        " Equity to Long-term Liability",
        " CFO to Assets",
        " Net Income to Total Assets",
        " Gross Profit to Sales",
        " Net Income to Stockholder's Equity",
        " Liability to Equity",
                    ] , axis=1)
        predictions = model.predict(data_df)
        return json.dumps({"result": predictions.tolist()})
    except Exception as e:
        return json.dumps({"error": str(e)})
