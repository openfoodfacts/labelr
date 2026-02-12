import typer


def check_label_studio_api_key(api_key: str | None):
    if not api_key:
        raise typer.BadParameter(
            "Label Studio API key not provided. Please provide it with the "
            "--api-key option or set the LABELR_LABEL_STUDIO_API_KEY environment variable."
        )
