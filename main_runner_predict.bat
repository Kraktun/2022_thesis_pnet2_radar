@echo off

call activate tf
:: setup project folder
set PROJECT_DIR=%~dp0
set PREPROC_NAME=NEW_DS_BASE_1
set LEARN_NAME=P-NET2_7_p
set MODEL_NAME=%PREPROC_NAME%_%LEARN_NAME%
set MODEL_IN_DIR=%PROJECT_DIR%\notebook_dumps\models_in\%MODEL_NAME%
:: run code
:: note: remove shuffle of dataset
echo STARTING MAIN
python main_predict.py --dataset-input %MODEL_IN_DIR%\%MODEL_NAME%.dataset.params.json --preproc-input %MODEL_IN_DIR%\%PREPROC_NAME%.preproc.params.json --model-input %MODEL_IN_DIR%\%LEARN_NAME%.model.params.json --train-input %MODEL_IN_DIR%\%MODEL_NAME%.train.params.json --load-model-name "%MODEL_IN_DIR%\*" -epochs 2 -batch 4 --preproc-prefix local --model-prefix local

pause