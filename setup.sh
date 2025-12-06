sudo yum install python3.12 -y -qq

rm -rf .env
python3.12 -m venv .env

source .env/bin/activate

pip install --upgrade pip -q
pip install -r python-requirements.txt -q
