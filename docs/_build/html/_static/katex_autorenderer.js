katex_options = {
displayMode: true, fleqn: true, macros: {"\\R":               "\\mathbb{R}",},
delimiters: [
        { left: "\\(", right: "\\)", display: false },
        { left: "\\[", right: "\\]", display: true }
        ]
}
document.addEventListener("DOMContentLoaded", function() {
  renderMathInElement(document.body, katex_options);
});
