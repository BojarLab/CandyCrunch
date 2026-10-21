from importlib.metadata import distribution, PackageNotFoundError
import json
import pathlib
import subprocess
import sys
import time
import warnings
package = distribution("candycrunch")
__version__ = package.version
# CandyCrunch's dev version follows glycowork's dev branch, whose version stays the same over many commits, so a stale glycowork is told by the
# commit it was installed from: GitHub is asked how many commits glycowork's dev has beyond it when that commit changed, when CandyCrunch's files
# changed (a pull or reinstall; at most hourly, as editing them changes them too), else once a day, and the answer is kept in ~/.candycrunch so that
# other imports stay offline. Only for CandyCrunch installed from git (it has a direct_url.json) or run from a checkout (whose egg-info has none),
# never for a PyPI release, whose users should stay on released glycowork
git_build = package.read_text("direct_url.json") or (pathlib.Path(__file__).parents[1] / ".git").exists()
if git_build and not getattr(sys, "frozen", False):  # the app ships the glycowork it was built with
    try:
        source = json.loads(distribution("glycowork").read_text("direct_url.json") or "{}")
        if "vcs_info" in source:
            commit = source["vcs_info"]["commit_id"]
            fix = 'pip install --force-reinstall --no-deps "glycowork @ git+https://github.com/BojarLab/glycowork.git@dev"'
        else:
            from urllib.request import url2pathname
            checkout = url2pathname(source["url"].removeprefix("file:"))
            commit = subprocess.run(["git", "-C", checkout, "rev-parse", "HEAD"], capture_output = True, text = True, check = True).stdout.strip()
            fix = f"git pull in {checkout}" + ("" if source["dir_info"].get("editable") else ", then pip install --no-deps it again")
        stamp = max(f.stat().st_mtime for f in pathlib.Path(__file__).parent.glob("*.py"))
        cache_file = pathlib.Path.home() / ".candycrunch" / "glycowork_check.json"
        cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
        age = time.time() - cache.get("checked", 0)
        if cache.get("commit") != commit or (cache.get("stamp") != stamp and age > 3600) or age > 86400:
            cache = {"commit": commit, "stamp": stamp, "checked": time.time(), "behind": 0}
            try:
                from urllib.request import urlopen
                with urlopen(f"https://api.github.com/repos/BojarLab/glycowork/compare/{commit}...dev?per_page=1", timeout = 3) as response:
                    cache["behind"] = json.load(response)["ahead_by"]
            except (OSError, ValueError, KeyError):  # offline, rate-limited, or a commit GitHub does not know (local, unpushed)
                pass
            cache_file.parent.mkdir(exist_ok = True)
            cache_file.write_text(json.dumps(cache))
        if cache["behind"]:
            warnings.warn(f"Your glycowork is {cache['behind']} commits behind glycowork's dev branch, which this CandyCrunch is developed against; "
                          f"update it with {fix}")
    except (PackageNotFoundError, KeyError, ValueError, OSError, subprocess.CalledProcessError):  # glycowork from PyPI or a source folder, no git,
        pass  # or a cache file another process is writing
