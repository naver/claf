mkdir data
echo "Make 'data' directory"

echo "Start download dataset from NSML"
nsml dataset pull reasoning-qa ./data

cd data
tar -xvpf reasoning-qa/train.tar

# move dataset and pretrained_vector
mv train/pretrained_vector ./
mv train/squad ./

rm -r reasoning-qa
rm -r train
