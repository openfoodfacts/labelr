from pathlib import Path
from typing import Annotated

import requests
import typer

app = typer.Typer(no_args_is_help=True)


DEFAULT_DIRECTUS_URL = "http://localhost:8055"


def _list_endpoint_iter(
    url: str,
    session: requests.Session,
    page_size: int,
    method: str = "GET",
    list_field: str | None = "data",
    **kwargs,
):
    """Iterate over paginated Directus endpoint.

    Args:
        url (str): URL of the Directus endpoint.
        session (requests.Session): Requests session to use for making HTTP
            requests.
        page_size (int): Number of items to fetch per page.
        method (str, optional): HTTP method to use. Defaults to "GET".
        list_field (str | None, optional): Field in the response JSON that
            contains the list of items. If None, the entire response is used as
            the list. Defaults to "data".
        **kwargs: Additional keyword arguments to pass to the requests method.
    Yields:
        dict: Items from the Directus endpoint.
    """
    page = 0
    next_page = True
    params = kwargs.pop("params", {})

    while next_page:
        params["offset"] = page * page_size
        params["limit"] = page_size
        r = session.request(method=method, url=url, params=params, **kwargs)
        r.raise_for_status()
        response = r.json()
        items = response[list_field] if list_field else response
        if len(items) > 0:
            yield from items
        else:
            next_page = False
        page += 1


def iter_items(
    collection_name: str,
    url: str,
    session: requests.Session,
    page_size: int = 50,
    **kwargs,
):
    """Iterate over items in a Directus collection.

    Args:
        collection_name (str): Name of the Directus collection.
        url (str): Base URL of the Directus server.
        session (requests.Session): Requests session to use for making HTTP
            requests.
        page_size (int, optional): Number of items to fetch per page. Defaults
            to 50.
        **kwargs: Additional keyword arguments to pass to the requests method.
    Yields:
        dict: Items from the Directus collection.
    """
    yield from _list_endpoint_iter(
        url=f"{url}/items/{collection_name}",
        session=session,
        page_size=page_size,
        **kwargs,
    )


@app.command()
def upload_data(
    dataset_path: Annotated[
        Path,
        typer.Option(
            help="Path to the dataset JSONL file to upload from.",
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    collection: Annotated[
        str, typer.Option(help="Name of the collection to upload the items to.")
    ],
    directus_url: Annotated[
        str,
        typer.Option(
            help="Base URL of the Directus server.",
        ),
    ] = DEFAULT_DIRECTUS_URL,
):
    """Upload data to a Directus collection."""
    import orjson
    import requests
    import tqdm

    session = requests.Session()

    with dataset_path.open("r") as f:
        for item in tqdm.tqdm(map(orjson.loads, f), desc="items"):
            r = session.post(
                f"{directus_url}/items/{collection}",
                json=item,
            )
            print(r.json())
            r.raise_for_status()


@app.command()
def update_items(
    collection: Annotated[
        str, typer.Option(help="Name of the collection to upload the items to.")
    ],
    directus_url: Annotated[
        str,
        typer.Option(
            help="Base URL of the Directus server.",
        ),
    ] = DEFAULT_DIRECTUS_URL,
    sort: Annotated[
        str | None,
        typer.Option(help="The field to sort items by, defaults to None (no sorting)."),
    ] = None,
    skip: Annotated[
        int, typer.Option(help="Number of items to skip, defaults to 0.")
    ] = 0,
):
    """Update items in a Directus collection.

    **Warning**: This command requires you to implement the processing
    function inside the command. It is provided as a template for batch
    updating items in a Directus collection.
    """
    import requests
    import tqdm

    session = requests.Session()

    params = {} if sort is None else {"sort[]": sort}
    for i, item in tqdm.tqdm(
        enumerate(
            iter_items(
                collection_name=collection,
                url=directus_url,
                session=session,
                params=params,
            )
        )
    ):
        if i < skip:
            typer.echo(f"Skipping item {i}")
            continue

        item_id = item["id"]
        # Implement your processing function here
        # It should return a dict with the fields to update only
        # If no update is needed, it should return None
        patch_item = None

        if patch_item is not None:
            r = session.patch(
                f"{directus_url}/items/{collection}/{item_id}",
                json=patch_item,
            )
            r.raise_for_status()


@app.command()
def export_data(
    output_path: Annotated[
        Path, typer.Option(help="Path to the file to export to.", allow_dash=True)
    ],
    collection: Annotated[
        str, typer.Option(help="Name of the collection to upload the items to.")
    ],
    directus_url: Annotated[
        str,
        typer.Option(
            help="Base URL of the Directus server.",
        ),
    ] = DEFAULT_DIRECTUS_URL,
):
    """Export a directus collection to a JSONL file."""
    import sys

    import orjson
    import requests
    import tqdm

    session = requests.Session()

    f = sys.stdout if output_path.as_posix() == "-" else output_path.open("w")
    with f:
        for item in tqdm.tqdm(
            iter_items(
                collection_name=collection,
                url=directus_url,
                session=session,
            )
        ):
            f.write(orjson.dumps(item).decode("utf-8") + "\n")
