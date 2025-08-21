COLORS = [
    "blue",
    "green",
    "yellow",
    "red",
    "purple",
    "orange",
    "pink",
    "brown",
    "gray",
    "black",
    "white",
]


def create_object_detection_label_config(labels_names: list[str]) -> str:
    """Create a Label Studio label configuration for object detection tasks.

    The format is the following:
    ```xml
    <View>
    <Image name="image" value="$image_url"/>
    <RectangleLabels name="label" toName="image">
    <Label value="nutrition-table" background="green"/>
        <Label value="nutrition-table-small" background="blue"/>
        <Label value="nutrition-table-small-energy" background="yellow"/>
        <Label value="nutrition-table-text" background="red"/>
    </RectangleLabels>
    </View>
    ```
    """
    if len(labels_names) > len(COLORS):
        raise ValueError(
            f"Too many labels ({len(labels_names)}) for the available colors ({len(COLORS)})."
        )
    labels_xml = "\n".join(
        f'    <Label value="{label}" background="{color}"/>'
        for label, color in zip(labels_names, COLORS[: len(labels_names)])
    )
    return f"""<View>
<Image name="image" value="$image_url"/>
<RectangleLabels name="label" toName="image">
{labels_xml}
</RectangleLabels>
</View>"""
