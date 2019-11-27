[ ! -f ../data/best.pth.zip ] && wget http://data.neu.ro/aclImdb.zip -O ../data/best.pth.zip
[ ! -d ../data/best.pth ] && unzip ../data/best.pth.zip -d ../data
