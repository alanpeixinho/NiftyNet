# -*- coding: utf-8 -*-
"""
Parse user configuration file
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import textwrap

from niftynet.engine.signal import TRAIN, INFER, EVAL, EXPORT
from niftynet.utilities.util_common import look_up_operations
from niftynet.engine.application_factory import ApplicationFactory
from niftynet.engine.application_factory import SUPPORTED_APP
from niftynet.io.misc_io import resolve_file_name
from niftynet.utilities.niftynet_global_config import NiftyNetGlobalConfig
from niftynet.utilities import NiftyNetLaunchConfig
from niftynet.utilities.user_parameters_custom import SUPPORTED_ARG_SECTIONS
from niftynet.utilities.user_parameters_custom import add_customised_args
from niftynet.utilities.user_parameters_default import \
    SUPPORTED_DEFAULT_SECTIONS
from niftynet.utilities.user_parameters_default import add_input_data_args
from niftynet.utilities.user_parameters_helper import has_section_in_config
from niftynet.utilities.user_parameters_helper import standardise_section_name
from niftynet.utilities.util_common import \
    damerau_levenshtein_distance as edit_distance
from niftynet.utilities.versioning import get_niftynet_version_string

SYSTEM_SECTIONS = {'SYSTEM', 'NETWORK', 'TRAINING', 'INFERENCE', 'EVALUATION'}
ACTIONS = {'train': TRAIN, 'inference': INFER, 'evaluation': EVAL, 'export': EXPORT}
EPILOG_STRING = \
    '\n\n======\nFor more information please visit:\n' \
    'http://niftynet.readthedocs.io/en/dev/config_spec.html\n' \
    '======\n\n'


# pylint: disable=protected-access
def available_keywords():
    """
    returns a list of all possible keywords defined in the parsers
    (duplicates from sections are removed.)
    """
    all_key_parser = argparse.ArgumentParser(
        parents=[],
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        conflict_handler='resolve')

    for _, add_args_func in SUPPORTED_DEFAULT_SECTIONS.items():
        all_key_parser = add_args_func(all_key_parser)

    all_key_parser = add_input_data_args(all_key_parser)

    # add keys from custom sections
    for _, add_args_func in SUPPORTED_ARG_SECTIONS.items():
        all_key_parser = add_args_func(all_key_parser)

    default_keys = []
    for action in all_key_parser._actions:
        try:
            default_keys.append(action.option_strings[0][2:])
        except (IndexError, AttributeError, ValueError):
            pass
    # remove duplicates
    default_keys = list(set(default_keys))
    # remove bad names
    default_keys = [keyword for keyword in default_keys if keyword]
    return default_keys


KEYWORDS = available_keywords()
NIFTYNET_HOME = NiftyNetGlobalConfig().get_niftynet_home_folder()


# pylint: disable=too-many-branches
#if no action parameter is specified, look for argv action command
def run(overwrite_args = {}, raise_error_unknown_args = True):
    """
    meta_parser is first used to find out location
    of the configuration file. Based on the application_name
    or meta_parser.prog name, the section parsers are organised
    to find system parameters and application specific
    parameters.

    :return: system parameters is a group of parameters including
        SYSTEM_SECTIONS and app_module.REQUIRED_CONFIG_SECTION
        input_data_args is a group of input data sources to be
        used by niftynet.io.ImageReader
    """
    meta_parser = argparse.ArgumentParser(
        description="Launch a NiftyNet application.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(EPILOG_STRING))
    version_string = get_niftynet_version_string()
    if overwrite_args.get('action', None) is None:
        meta_parser.add_argument("action",
                                 help="train networks, run inferences "
                                      "or evaluate inferences",
                                 metavar='ACTION',
                                 choices=list(ACTIONS))
    meta_parser.add_argument("-v", "--version",
                             action='version',
                             version=version_string)
    if overwrite_args.get('config_file', None) is None:
        meta_parser.add_argument("-c", "--conf",
                                 help="specify configurations from a file",
                                 metavar="CONFIG_FILE")
    if overwrite_args.get('app', None) is None:
        meta_parser.add_argument("-a", "--application_name",
                                 help="specify an application module name",
                                 metavar='APPLICATION_NAME',
                                 default="")

    meta_args, args_from_cmdline = meta_parser.parse_known_args()
    print(version_string)

    # read configurations, to be parsed by sections
    if overwrite_args.get('config_file', None) is None:
        config_file_name = __resolve_config_file_path(meta_args.conf)
    else:
        config_file_name = __resolve_config_file_path(overwrite_args['config_file'])
    config = NiftyNetLaunchConfig()
    config.read([config_file_name])

    # infer application name from command
    app_name = None
    try:
        pname = overwrite_args.get('app', meta_parser.prog)
        parser_prog = pname.replace('.py', '')
        app_name = parser_prog if parser_prog in SUPPORTED_APP \
            else meta_args.application_name
        assert app_name
    except (AttributeError, AssertionError):
        raise ValueError(
            "\nUnknown application {}, or did you forget '-a' "
            "command argument?{}".format(app_name, EPILOG_STRING))

    # load application by name
    app_module = ApplicationFactory.create(app_name)
    try:
        assert app_module.REQUIRED_CONFIG_SECTION, \
            "\nREQUIRED_CONFIG_SECTION should be static variable " \
            "in {}".format(app_module)
        has_section_in_config(config, app_module.REQUIRED_CONFIG_SECTION)
    except AttributeError:
        raise AttributeError(
            "Application code doesn't have REQUIRED_CONFIG_SECTION property. "
            "{} should be an instance of "
            "niftynet.application.base_application".format(app_module))
    except ValueError:
        raise ValueError(
            "\n{} requires [{}] section in the config file.{}".format(
                app_name, app_module.REQUIRED_CONFIG_SECTION, EPILOG_STRING))

    # check keywords in configuration file
    _check_config_file_keywords(config)

    # using configuration as default, and parsing all command line arguments
    # command line args override the configure file options
    all_args = {}
    for section in config.sections():
        # try to rename user-specified sections for consistency
        section = standardise_section_name(config, section)
        section_defaults = dict(config.items(section))
        section_args, args_from_cmdline = \
            _parse_arguments_by_section([],
                                        section,
                                        section_defaults,
                                        args_from_cmdline,
                                        app_module.REQUIRED_CONFIG_SECTION)
        all_args[section] = section_args
        dict_args = vars(section_args)
        overwrite_args_keys = overwrite_args[section.lower()].keys() if section.lower() in overwrite_args else None


        if overwrite_args_keys is not None:
            #if there is an overwriten argument passed to the function
            #ignore both config file and argv
            for arg_key in dict_args:
                if arg_key in overwrite_args_keys:
                    dict_args[arg_key] = overwrite_args[section.lower()][arg_key]

    # check if any args from command line not recognised
    if raise_error_unknown_args:
        _check_cmd_remaining_keywords(list(args_from_cmdline))

    # split parsed results in all_args
    # into dictionaries of system_args and input_data_args
    system_args, input_data_args = {}, {}
    for section in all_args:

        # copy system default sections to ``system_args``
        if section in SYSTEM_SECTIONS:
            system_args[section] = all_args[section]
            continue

        # copy application specific sections to ``system_args``
        if section == app_module.REQUIRED_CONFIG_SECTION:
            system_args['CUSTOM'] = all_args[section]
            vars(system_args['CUSTOM'])['name'] = app_name
            continue

        # copy non-default sections to ``input_data_args``
        input_data_args[section] = all_args[section]

        # set the output path of csv list if not exists
        try:
            csv_path = resolve_file_name(
                input_data_args[section].csv_file,
                (os.path.dirname(config_file_name), NIFTYNET_HOME))
            input_data_args[section].csv_file = csv_path
            # don't search files if csv specified in config
            try:
                delattr(input_data_args[section], 'path_to_search')
            except AttributeError:
                pass
        except (IOError, TypeError):
            input_data_args[section].csv_file = ''

    # preserve ``config_file`` and ``action parameter`` from the meta_args
    system_args['CONFIG_FILE'] = argparse.Namespace(path=config_file_name)
    # mapping the captured action argument to a string in ACTIONS
    if overwrite_args.get('action', None) is None:
        system_args['SYSTEM'].action = \
            look_up_operations(meta_args.action, ACTIONS)
    else:
        system_args['SYSTEM'].action = look_up_operations(overwrite_args['action'], ACTIONS)
    if overwrite_args.get('model_dir', None) is None:
        if not system_args['SYSTEM'].model_dir:
            system_args['SYSTEM'].model_dir = os.path.join(
                os.path.dirname(config_file_name), 'model')
    else:
        system_args['SYSTEM'].model_dir = overwrite_args['model_dir']
    return system_args, input_data_args


def _parse_arguments_by_section(parents,
                                section,
                                args_from_config_file,
                                args_from_cmd,
                                required_section):
    """
    This function first adds parameter names to a parser,
    according to the section name.
    Then it loads values from configuration files as tentative params.
    Finally it overrides existing pairs of 'name, value' with commandline
    inputs.

    Commandline inputs only override system/custom parameters.
    input data related parameters needs to be defined in config file.

    :param parents: a list, parsers will be created as
        subparsers of parents
    :param section: section name to be parsed
    :param args_from_config_file: loaded parameters from config file
    :param args_from_cmd: dictionary commandline parameters
    :return: parsed parameters of the section and unknown
        commandline params.
    """
    #import pdb; pdb.set_trace()
    section_parser = argparse.ArgumentParser(
        parents=parents,
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    try:
        add_args_func = SUPPORTED_DEFAULT_SECTIONS[section]
    except KeyError:
        if section == required_section:
            def add_args_func(parser):
                """
                wrapper around add_customised_args
                """
                return add_customised_args(parser, section.upper())
        else:
            # all remaining sections are defaulting to input section
            add_args_func = add_input_data_args

    section_parser = add_args_func(section_parser)

    # loading all parameters a config file first
    if args_from_config_file is not None:
        section_parser.set_defaults(**args_from_config_file)
    # input command line input overrides config file
    if (section in SYSTEM_SECTIONS) or (section == required_section):
        section_args, unknown = section_parser.parse_known_args(args_from_cmd)
        return section_args, unknown
    # don't parse user cmd for input source sections
    section_args, _ = section_parser.parse_known_args([])
    return section_args, args_from_cmd


def _check_config_file_keywords(config):
    """
    check config files, validate keywords provided against
    parsers' argument list
    """

    # collecting all keywords from the config
    config_keywords = []
    for section in config.sections():
        if config.items(section):
            config_keywords.extend(list(dict(config.items(section))))
    _raises_bad_keys(config_keywords, error_info='config file')


def _check_cmd_remaining_keywords(args_from_cmdline):
    """
    check list of remaining arguments from the command line input.
    Normally `args_from_cmd` should be empty; non-empty list
    means unrecognised parameters.
    """

    args_from_cmdline = [
        arg_item.replace('-', '') for arg_item in args_from_cmdline]
    _raises_bad_keys(args_from_cmdline, error_info='command line')

    # command line parameters should be valid
    # assertion will be triggered when keywords matched ones in custom
    # sections that are not used in the current application.
    assert not args_from_cmdline, \
        '\nUnknown parameter: {}{}'.format(args_from_cmdline, EPILOG_STRING)


def _raises_bad_keys(keys, error_info='config file'):
    """
    raises value error if keys is not in the system key set.
    `error_info` is used to customise the error message.
    """
    for key in list(keys):
        if key in KEYWORDS:
            continue
        dists = {k: edit_distance(k, key) for k in KEYWORDS}
        closest = min(dists, key=dists.get)
        raise ValueError(
            'Unknown keywords in {3}: By "{0}" '
            'did you mean "{1}"?\n "{0}" is '
            'not a valid option.{2}'.format(
                key, closest, EPILOG_STRING, error_info))


def __resolve_config_file_path(cmdline_arg):
    """
    Search for the absolute file name of the configuration file.
    starting from `-c` value provided by the user.

    :param cmdline_arg:
    :return:
    """
    if not cmdline_arg:
        raise IOError("\nNo configuration file has been provided, did you "
                      "forget '-c' command argument?{}".format(EPILOG_STRING))
    # Resolve relative configuration file location
    config_file_path = os.path.expanduser(cmdline_arg)
    try:
        config_file_path = resolve_file_name(
            config_file_path, ('.', NIFTYNET_HOME))
        if os.path.isfile(config_file_path):
            return config_file_path
    except (IOError, TypeError):
        config_file_path = os.path.expanduser(cmdline_arg)

    config_file_path = os.path.join(
        NiftyNetGlobalConfig().get_default_examples_folder(),
        config_file_path, config_file_path + "_config.ini")
    if os.path.isfile(config_file_path):
        return config_file_path

    # could not proceed without a configuration file
    raise IOError("\nConfiguration file not found: {}.{}".format(
        os.path.expanduser(cmdline_arg), EPILOG_STRING))
