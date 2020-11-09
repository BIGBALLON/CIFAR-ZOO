echo "Running isort..."
isort -y -sp ./dev

echo "Running black..."
black .

echo "Running flake8..."
flake8 . --config ./dev/setup.cfg