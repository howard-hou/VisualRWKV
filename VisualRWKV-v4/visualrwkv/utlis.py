def postprocess_response(decoded, stop_token="\n\n"):
    # remove the stop token
    processed = []
    for d in decoded:
        d = d.split(stop_token)[0]
        processed.append(d)
    return processed
