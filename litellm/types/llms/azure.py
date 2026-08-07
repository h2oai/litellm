API_VERSION_YEAR_SUPPORTED_RESPONSE_FORMAT = 2024
API_VERSION_MONTH_SUPPORTED_RESPONSE_FORMAT = 8

# First api_version year in which Azure chat completions reject `max_tokens`
# in favour of `max_completion_tokens`:
#
#     Unsupported parameter: 'max_tokens' is not supported with this model.
#     Use 'max_completion_tokens' instead.
#
# Observed on 2025-04-01-preview against a gpt-4o deployment, i.e. for a plain
# chat model and not only for the o-series. The boundary is set at the year
# rather than a specific preview date because Azure rolled the change out
# across the 2025 preview versions and the v1 (`preview` / `latest` / `v1`)
# API, and because `max_completion_tokens` is the field Azure documents as
# current for every 2025 version — so erring on this side of the boundary
# sends the field Azure asks for.
API_VERSION_YEAR_REQUIRING_MAX_COMPLETION_TOKENS = 2025
