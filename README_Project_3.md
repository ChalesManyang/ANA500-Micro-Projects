# ANA500 Micro-Project 3: Airline Arrival-Delay Regression
Charles Ayuk | National University

Repository: https://github.com/ChalesManyang/ANA500-Micro-Projects

This project follows the supplied template through Act and compares ordinary least squares with linear support vector regression. Its random splits match Project 4.

Use Python 3.11+, install requirements.txt, and put the supplied airline.csv at the repository root or beside the notebook. Run cells in order. The notebook contains executed results and figures. The Python script contains the same commented code, and the compiled code PDF includes code and outputs.

Both regression models achieve test RMSE of approximately 10.03 minutes. Carrying departure delay forward produces the lowest test MAE (5.04 minutes). The data lacks flight/date identifiers, so random passenger holdouts cannot establish future-flight performance. Tuning and model selection use validation data before test evaluation.
