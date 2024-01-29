#Run sequential noisy and denoised training scripts in parallel
printf 'Starting full run of training-saving-counterfactuals-retraining-evaluating of all variants'
start=`date +%s`

#Train denoised and noisy models on training data
printf '\nTraining models on original training set'
python BNN/BNN.py --TRAIN &
python BNN/BNN.py --TRAIN --NOISY &

wait

end=`date +%s`
runtime=$((end-start))

printf "\nTraining completed. Runtime: $runtime seconds"

#Save results to .json files to be used for counterfactuals
printf '\nSaving results to .json files'
python BNN/BNN.py --SAVE &
python BNN/BNN.py --SAVE --NOISY &

wait

end=`date +%s`
runtime=$((end-start))

printf "\nFiles saved. Runtime: $runtime seconds"

#Convert files to counterfactuals
printf '\nGenerating counterfactuals'
python DiCE_uncertainty/DiCE_uncertainty.py $
python DiCE_uncertainty/DiCE_uncertainty.py --NOISY $

wait

end=`date +%s`
runtime=$((end-start))

printf "\nCounterfactuals generated. Runtime: $runtime seconds"

#Retrain models with noisy, denoised and original data
printf '\nRetraining models with/without counterfactuals'
python BNN/BNN.py --TRAIN --CF_TRAIN &
python BNN/BNN.py --TRAIN --NOISY --CF_TRAIN 
python BNN/BNN.py --TRAIN --NOCF_TRAIN &
python BNN/BNN.py --TRAIN --NOISY --NOCF_TRAIN &

wait

end=`date +%s`
runtime=$((end-start))

printf "\nModels retrained. Runtime: $runtime seconds"

#Evaluate and save models
printf '\nEvaluating all model variantions and saving to .json files'
python BNN/BNN.py --SAVE --EVAL &
python BNN/BNN.py --SAVE --EVAL --NOISY &
python BNN/BNN.py --SAVE --EVAL --CF_TRAIN &
python BNN/BNN.py --SAVE --EVAL --NOISY --CF_TRAIN &
python BNN/BNN.py --SAVE --EVAL --NOCF_TRAIN &
python BNN/BNN.py --SAVE --EVAL --NOISY --NOCF_TRAIN &

printf '\nFiles saved'

end=`date +%s`
runtime=$((end-start))

printf "\nRun completed. Runtime: $runtime seconds"