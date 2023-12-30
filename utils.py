import toml

def get_api_keys(api_config_file:str = 'api.toml'):
    with open(api_config_file) as f:
        api_keys = toml.load(f)['api_keys']
    return api_keys
