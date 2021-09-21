if [ ! -f models/KneeNet.0 ]; then
    mkdir -p models
    mkdir -p output
    curl https://s3-eu-west-1.amazonaws.com/kidzinski/models/KneeNet/KneeNet.0 -o models/KneeNet.0 
fi
docker run \
    -v ${PWD}/input:/workspace/input \
    -v ${PWD}/output:/workspace/output \
    -v ${PWD}/scripts:/workspace/scripts \
    -v ${PWD}/models:/workspace/models \
    -it kidzik/kneenet:latest \
    python scripts/predict.py && printf "\n-- RESULTS (in output/prediction.csv) --\n" && cat output/predictions.csv | column -t -s,
