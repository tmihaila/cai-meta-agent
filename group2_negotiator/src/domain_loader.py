import importlib.util
from pathlib import Path


def load_domain(domain_path: Path):
    spec = importlib.util.spec_from_file_location(domain_path.stem, domain_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.issues, module.ufun_a, module.ufun_b


def list_domains(domains_dir: Path):
    return sorted(domains_dir.glob("domain*.py"))
