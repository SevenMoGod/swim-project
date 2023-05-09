

def print_runtime(total, model_init, image_load, grounddino, sam, postprocess):
    print(
        "Total runtime: {:.2f}s | Model init: {:.2f}s | Image load: {:.2f}s | GroundDINO: {:.2f}s | SAM: {:.2f}s | Postprocess: {:.2f}s".format(
            total, model_init, image_load, grounddino, sam, postprocess))
