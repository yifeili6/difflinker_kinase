from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from kinase_gap.stereo import install
from kinase_gap.generate import main
install()
if __name__ == '__main__':
    raise SystemExit(main())
