from sklearn.utils._tags import _safe_tags

from imblearn.pipeline import Pipeline


class ImbalancedPipeline(Pipeline):
    def _more_tags(self):
        # get tags from the pipeline's final step
        step_last = self.steps[-1]
        pipe_tags = step_last[-1]._get_tags()

        # check if all steps allow NaNs
        tag_allow_nan = True
        for step in self.steps:
            step_tag_allow_nan = _safe_tags(step[-1], 'allow_nan')
            tag_allow_nan = (tag_allow_nan & step_tag_allow_nan)
        pipe_tags['allow_nan'] = tag_allow_nan

        return pipe_tags
