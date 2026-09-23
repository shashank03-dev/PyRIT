# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Literal, get_args

from pyrit.converter.converter import Converter, ConverterResult
from pyrit.models import ComponentIdentifier, PromptDataType

#: The markup techniques used to hide the payload inside a carrier document.
HiddenTextTechnique = Literal[
    "html_comment",
    "css_display_none",
    "css_offscreen",
    "html_alt_text",
    "aria_label",
    "markdown_comment",
    "markdown_link_title",
]


class HiddenTextConverter(Converter):
    """
    Embeds the prompt as hidden text inside an HTML or Markdown carrier document.

    This converter supports testing for *indirect prompt injection* (also called cross-domain
    prompt injection, or XPIA). Systems that summarize or act on external content, such as web
    pages, documents, or emails, ingest the raw source of that content. Markup allows text to be
    present in the source while remaining invisible or de-emphasized to a human reviewer, e.g.
    inside an HTML comment, a ``display:none`` element, or an image ``alt`` attribute. A model
    that parses the raw source may still read and act on that hidden text.

    The converter wraps the incoming prompt (the injected instruction) using the selected
    ``technique`` and, optionally, surrounds it with benign ``carrier_text`` so the payload is
    embedded within realistic-looking content. It only formats the text it is given; it does not
    generate any attack content on its own.

    This mirrors existing payload-formatting converters such as ``AnsiAttackConverter`` (terminal
    escape sequences) and ``TransparencyAttackConverter`` (imperceptible image layers), extending
    the same idea to HTML and Markdown documents. It is intended to be used as a request converter
    with the XPIA workflow (see ``pyrit.executor.workflow.xpia``).

    The hiding techniques used here are documented indirect-injection vectors studied in web-agent
    security research such as the WASP benchmark [@evtimov2025wasp].
    """

    SUPPORTED_INPUT_TYPES = ("text",)
    SUPPORTED_OUTPUT_TYPES = ("text",)

    def __init__(
        self,
        *,
        technique: HiddenTextTechnique = "html_comment",
        carrier_text: str = "",
    ) -> None:
        """
        Initialize the converter with a hiding technique and optional carrier text.

        Args:
            technique (HiddenTextTechnique): The markup technique used to hide the payload.
                Defaults to ``"html_comment"``. One of:

                - ``"html_comment"``: wrap the payload in an ``<!-- ... -->`` HTML comment.
                - ``"css_display_none"``: place the payload in a ``<span>`` styled with
                  ``display:none``.
                - ``"css_offscreen"``: place the payload in a ``<span>`` positioned off-screen.
                - ``"html_alt_text"``: place the payload in the ``alt`` attribute of an
                  ``<img>`` tag.
                - ``"aria_label"``: place the payload in the ``aria-label`` attribute of a
                  ``<span>``.
                - ``"markdown_comment"``: place the payload in a Markdown comment
                  (``[//]: # (...)``).
                - ``"markdown_link_title"``: place the payload in the title of a Markdown link.
            carrier_text (str): Optional visible text that surrounds the hidden payload so it is
                embedded within realistic content. When empty (the default), only the hidden markup
                is produced.

        Raises:
            ValueError: If ``technique`` is not a supported technique.
        """
        super().__init__()
        if technique not in get_args(HiddenTextTechnique):
            raise ValueError(
                f"Unsupported technique '{technique}'. Supported techniques: {list(get_args(HiddenTextTechnique))}"
            )
        self._technique = technique
        self._carrier_text = carrier_text

    def _build_identifier(self) -> ComponentIdentifier:
        """
        Build the converter identifier with the hiding technique and carrier configuration.

        The technique and whether carrier text is present are included so that converters with
        different configurations produce distinct identifiers.

        Returns:
            ComponentIdentifier: The identifier for this converter.
        """
        return self._create_identifier(
            params={
                "technique": self._technique,
                "has_carrier_text": bool(self._carrier_text),
            }
        )

    def _hide(self, payload: str) -> str:
        """
        Wrap the payload in the markup for the configured technique.

        Args:
            payload (str): The text to hide.

        Returns:
            str: The payload wrapped in the selected hiding markup.
        """
        if self._technique == "html_comment":
            # HTML comments are stripped from the rendered page but remain in the source.
            safe = payload.replace("--", "––")
            return f"<!-- {safe} -->"
        if self._technique == "css_display_none":
            safe = payload.replace("<", "&lt;").replace(">", "&gt;")
            return f'<span style="display:none">{safe}</span>'
        if self._technique == "css_offscreen":
            safe = payload.replace("<", "&lt;").replace(">", "&gt;")
            return f'<span style="position:absolute;left:-9999px;top:-9999px">{safe}</span>'
        if self._technique == "html_alt_text":
            safe = payload.replace('"', "&quot;")
            return f'<img src="x" alt="{safe}">'
        if self._technique == "aria_label":
            safe = payload.replace('"', "&quot;")
            return f'<span aria-label="{safe}"></span>'
        if self._technique == "markdown_comment":
            # Markdown link-reference comment syntax; not rendered by Markdown processors.
            safe = payload.replace("\n", " ")
            return f"[//]: # ({safe})"
        # markdown_link_title
        safe = payload.replace('"', "'")
        return f'[ ](https://example.com "{safe}")'

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Convert the given prompt by embedding it as hidden text in a carrier document.

        Args:
            prompt (str): The prompt to hide (the injected instruction).
            input_type (PromptDataType): The type of input data. Only ``text`` is supported.

        Returns:
            ConverterResult: The result containing the carrier document with the hidden payload.

        Raises:
            ValueError: If the input type is not supported.
        """
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")

        hidden = self._hide(prompt)
        output_text = f"{self._carrier_text}\n{hidden}" if self._carrier_text else hidden

        return ConverterResult(output_text=output_text, output_type="text")
