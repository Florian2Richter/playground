import sys
import subprocess
from pathlib import Path


def test_fizz_buzz_first_15():
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'fizz_buzz.py'
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    lines = result.stdout.strip().splitlines()[:15]
    expected = [
        '1',
        '2',
        'Fizz!',
        '4',
        'Buzz!',
        'Fizz!',
        '7',
        '8',
        'Fizz!',
        'Buzz!',
        '11',
        'Fizz!',
        '13',
        '14',
        'FizzBuzz!',
    ]
    assert lines == expected
