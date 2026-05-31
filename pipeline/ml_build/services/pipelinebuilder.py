from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from ml_build.config_loader import load_config
from ml_build.logger import get_logger

log = get_logger("Pipelinebuilder")

def building_pipeline():
    try:
        conf = load_config()
        log.info("loading the cv and the scoring parameters from the config file")
        c_v = conf['parameter_grid']['cv']
        score = conf['parameter_grid']['scoring']
        if c_v is None or score is None:
            raise ValueError("Cross-validation and scoring parameters must be defined in the configuration.")
        log.info("Successfully loaded the parameters from the config file")
        log.info("Building the Random Forest pipeline")
        RandomForest_pipeline = Pipeline(steps=[('model', RandomForestClassifier(random_state=42))])
        log.info("Successfully built the Random Forest pipeline")
        log.info("loading the parameter grid configuration for the grid search")
        nestimators = conf['parameter_grid']['n_estimators']
        maxdepth = conf['parameter_grid']['max_depth']
        minsamplesplit = conf['parameter_grid']['min_samples_split']
        minsamplesleaf = conf['parameter_grid']['min_samples_leaf']
        if nestimators is None or maxdepth is None or minsamplesplit is None or minsamplesleaf is None:
            raise ValueError("All parameters for the grid search must be defined in the configuration.")
        log.info("Successfully loaded the parameter grid configuration")
        log.info("Setting up the GridSearchCV for the Random Forest pipeline")
        RandomForest_paramgrid = {
            'model__n_estimators': nestimators,
            'model__max_depth': maxdepth,
            'model__min_samples_split': minsamplesplit,
            'model__min_samples_leaf': minsamplesleaf
            }

        log.info("Successfully set up the GridSearchCV for the Random Forest pipeline")
        RandomForest_grid = GridSearchCV(estimator=RandomForest_pipeline, param_grid=RandomForest_paramgrid, cv=c_v, scoring=score)
        log.info("Successfully created the GridSearchCV for the Random Forest pipeline")
        return RandomForest_grid
    except Exception as e:
        log.error(f"Error during pipeline building: {e}")
        return None
    
