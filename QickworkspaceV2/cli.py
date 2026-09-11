"""Read-only project tools and hardware measurement/service entry points."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(prog="qickworkspace")
    parser.add_argument("--project", default="lab/project.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate")
    sub.add_parser("catalog")
    run = sub.add_parser("run")
    run.add_argument("experiment")
    run.add_argument("--target", default="Q1")
    run.add_argument("--parameters", default="{}", help="JSON object")
    serve = sub.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    from QickworkspaceV2 import Session

    from QickworkspaceV2.device.models import Device, ProjectConfig, read_yaml
    from QickworkspaceV2.experiments import default_registry
    from importlib import import_module

    path = Path(args.project).resolve()
    project = ProjectConfig.model_validate(read_yaml(path))
    device = Device.from_files(path.parent / project.hardware, path.parent / project.device)
    registry = default_registry()
    for module in project.experiment_modules:
        registry.register(import_module(module).experiment)
    if args.command == "validate":
        print(json.dumps(device.summary(), indent=2, ensure_ascii=False))
    elif args.command == "catalog":
        from QickworkspaceV2.runtime.catalog import catalog_payload

        print(json.dumps(catalog_payload(registry), indent=2))
    elif args.command == "run":
        session = Session.from_project(args.project)
        result = session.run(args.experiment, target=args.target, **json.loads(args.parameters))
        print(
            json.dumps(
                {"run_id": result.run_id, "quality": result.quality, "metrics": result.metrics}, indent=2
            )
        )
    elif args.command == "serve":
        import uvicorn
        from QickworkspaceV2.runtime.service import create_app

        uvicorn.run(create_app(Session.from_project(args.project)), host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
