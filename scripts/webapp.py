import streamlit as st
import shlex
import os
import re
import sys
import copy
import warnings
import time
import ldm.dream.readline
from ldm.dream.pngwriter import PngWriter, PromptFormatter
from ldm.dream.server import DreamServer, ThreadingDreamServer
from ldm.dream.image_util import make_grid
from omegaconf import OmegaConf


def main(outdir):
    """Initialize command-line parsers and the diffusion model"""

    # models = OmegaConf.load("configs/models.yaml") (PROBABLY NOT NEEDED)
    width = 512
    height = 512
    config = R"..\configs\stable-diffusion\v1-inference.yaml"
    weights = R"..\models\ldm\stable-diffusion-v1\model.ckpt"

    print('* Initializing, be patient...\n')
    sys.path.append('.')
    from pytorch_lightning import logging
    from ldm.generate import Generate

    # these two lines prevent a horrible warning message from appearing
    # when the frozen CLIP tokenizer is imported
    import transformers

    transformers.logging.set_verbosity_error()

    # creating a simple text2image object with a handful of
    # defaults passed on the command line.
    # additional parameters will be added (or overriden) during
    # the user input loop
    t2i = Generate(
        width=width,
        height=height,
        sampler_name="k_lms",
        weights=weights,
        full_precision=True,
        config=config,
        grid=False,
        # this is solely for recreating the prompt
        seamless=False,
        # embedding_path="",
        # device_type=torch.cuda.current_device(),
        ignore_ctrl_c=False,
    )

    # make sure the output directory exists
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # gets rid of annoying messages about random seed
    logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)

    # preload the model
    t2i.load_model()

    print(
        "\n* Initialization done!"
        )
    return t2i
    

def generate_image(t2i, outdir, prompt):

    path_filter = re.compile(r'[<>:"/\\|?*]')
    last_results = list()


    if len(prompt) == 0:
        return print('Try again with a prompt!')
        
    current_outdir = outdir

    # Here is where the images are actually generated!
    last_results = []
    try:
        file_writer = PngWriter(current_outdir)
        prefix = file_writer.unique_prefix()
        results = []  # list of filename, prompt pairs

        def image_writer(image, seed, upscaled=False):
            path = None
            filename = f'{prefix}.{seed}.png'
            
            # normalized_prompt = PromptFormatter(
            #     t2i, prompt).normalize_prompt()
            # metadata_prompt = f'{normalized_prompt} -S{seed}'

            path = file_writer.save_image_and_prompt_to_png(
                image, prompt, filename)

            # if (not upscaled):
            #     # only append to results if we didn't overwrite an earlier output
            #     results.append([path, metadata_prompt])
            # last_results.append([path, seed])

        prompt_dict = {'prompt': prompt}
        t2i.prompt2image(image_callback=image_writer, **prompt_dict)

    except AssertionError as e:
        print(e)
        return

    except OSError as e:
        print(e)
        return

    print('Outputs:')
    log_path = os.path.join(current_outdir, 'dream_log.txt')
    write_log_message(results, log_path)
    print()

    print('goodbye!')


def write_log_message(results, log_path):
    """logs the name of the output image, prompt, and prompt args to the terminal and log file"""
    global output_cntr
    log_lines = [f'{path}: {prompt}\n' for path, prompt in results]
    for l in log_lines:
        output_cntr += 1
        print(f'[{output_cntr}] {l}',end='')


    with open(log_path, 'a', encoding='utf-8') as file:
        file.writelines(log_lines)


outdir = R"C:\Users\tobia\Documents\Coding\projects\stable_diffusion\dummy-repo\outputs\img-samples"

if 't2i' not in globals():
    t2i = main(outdir) 

prompt = st.text_input(label= "Type in your prompt.")

if st.button("Create Image"):
    generate_image(t2i, outdir, prompt)