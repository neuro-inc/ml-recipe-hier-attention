DATA_PATH=/data

mkdir -p ${DATA_PATH}
wget -q -nc http://data.neu.ro/aclImdb.zip -O /tmp/aclImdb.zip
unzip -n -qq /tmp/aclImdb.zip -d ${DATA_PATH}
