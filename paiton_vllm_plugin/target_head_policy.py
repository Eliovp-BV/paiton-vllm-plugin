"""Conservative host policy for the experimental approximate greedy head."""


def permits_sparse_greedy_head(params):
    if params is None:
        return False
    required = dict(n=1, temperature=0.0, presence_penalty=0.0,
                    frequency_penalty=0.0, repetition_penalty=1.0, min_tokens=0)
    if any(getattr(params, key, None) != value for key, value in required.items()):
        return False
    # These features consume or modify the full vocabulary, or introduce a
    # processor whose support is not represented by the approximate shortlist.
    absent = ('logprobs', 'prompt_logprobs', 'logprob_token_ids', 'structured_outputs',
              'logit_bias', 'allowed_token_ids', 'extra_args',
              'thinking_token_budget',
              'repetition_detection')
    # vLLM normalizes an absent bad_words setting to an empty list.
    empty = ('bad_words', '_bad_words_token_ids', 'logits_processors')
    return (all(getattr(params, key, None) is None for key in absent)
            and all(not getattr(params, key, None) for key in empty))


class GreedyBatchPolicy:
    def __init__(self):
        self.requests = {}
        self.current_allowed = False

    def update(self, scheduler_output):
        self.current_allowed = False
        for request_id in scheduler_output.finished_req_ids:
            self.requests.pop(request_id, None)
        for request in scheduler_output.scheduled_new_reqs:
            self.requests[request.req_id] = permits_sparse_greedy_head(request.sampling_params)
        scheduled = scheduler_output.num_scheduled_tokens
        self.current_allowed = (
            bool(scheduled) and not scheduler_output.has_structured_output_requests
            and all(self.requests.get(request_id, False) for request_id in scheduled)
        )

    def allows_sampling(self, grammar_output):
        return self.current_allowed and grammar_output is None
