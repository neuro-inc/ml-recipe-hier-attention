set -e

DATA_PATH=${1:-../data}

echo "Downloading dataset to $DATA_PATH"
mkdir -p $DATA_PATH
wget -nc http://data.neu.ro/aclImdb.zip -O /tmp/aclImdb.zip
echo "Unpacking..."
unzip -n -q /tmp/aclImdb.zip -d $DATA_PATH
echo "Finished"