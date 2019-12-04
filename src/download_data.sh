DATA_PATH=../data

mkdir -p ${DATA_PATH}
wget -nc http://data.neu.ro/aclImdb.zip -O ${DATA_PATH}/aclImdb.zip
unzip -n -qq ${DATA_PATH}/aclImdb.zip -d ${DATA_PATH}
