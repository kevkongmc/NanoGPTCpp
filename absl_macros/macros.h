#ifndef NANOGPT_CPP_ABSL_MACROS_MACROS_H_
#define NANOGPT_CPP_ABSL_MACROS_MACROS_H_

// Evaluates an expression that produces a `absl::Status` or `absl::StatusOr<T>`.
// If the status is not OK, returns the status from the current function.
//
// Example usage:
//   RETURN_IF_ERROR(DoSomething());
//   RETURN_IF_ERROR(DoSomethingElse()) << "Additional context";
#define RETURN_IF_ERROR(expr)                        \
  do {                                               \
    /* Using _status to avoid shadowing warnings */  \
    const auto _status = (expr);                     \
    if (!_status.ok()) return _status;               \
  } while (0)

// Assigns the result of an expression to a variable, or returns on error.
//
// Example usage:
//   ASSIGN_OR_RETURN(auto value, MaybeGetValue());
//   ASSIGN_OR_RETURN(const auto& value, MaybeGetValue(),
//                    _ << "Additional context");
#define ASSIGN_OR_RETURN(lhs, expr, ...) \
  ASSIGN_OR_RETURN_IMPL(                 \
      NANOGPT_STATUS_MACROS_IMPL_CONCAT(_status_or_value, __LINE__), \
      lhs, expr, __VA_ARGS__)

// Internal helper for ASSIGN_OR_RETURN.
#define ASSIGN_OR_RETURN_IMPL(statusor, lhs, expr, ...)  \
  auto statusor = (expr);                                \
  if (!statusor.ok()) {                                  \
    return ::absl::Status(                               \
        statusor.status().code(),                        \
        statusor.status().message()                      \
            << (__VA_ARGS__ + 0));                       \
  }                                                      \
  lhs = std::move(statusor).value();

// Helper macros for token concatenation
#define NANOGPT_STATUS_MACROS_IMPL_CONCAT_INNER(x, y) x##y
#define NANOGPT_STATUS_MACROS_IMPL_CONCAT(x, y) \
  NANOGPT_STATUS_MACROS_IMPL_CONCAT_INNER(x, y)

#endif  // NANOGPT_CPP_ABSL_MACROS_MACROS_H_
