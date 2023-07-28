def _impl(repository_ctx):
    download_url = "https://download.pytorch.org/libtorch/cpu/libtorch-macos-2.0.1.zip"
    repository_ctx.download_and_extract(download_url, stripPrefix = "libtorch")

    repository_ctx.template(
        "BUILD",
        repository_ctx.path(Label(repository_ctx.attr.build_file_template)),
    )

libtorch_repository = repository_rule(
    implementation = _impl,
    attrs = {"build_file_template": attr.string(mandatory = True)},
)
