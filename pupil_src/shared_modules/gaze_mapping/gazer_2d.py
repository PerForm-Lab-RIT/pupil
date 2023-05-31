"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2021 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import abc
import logging
import typing as T
import numpy as np
from sklearn.linear_model import LinearRegression

from methods import normalize
from .gazer_3d.utils import _clamp_norm_point

from gaze_mapping.gazer_base import (
    GazerBase,
    Model,
    NotEnoughDataError,
)


logger = logging.getLogger(__name__)


_REFERENCE_FEATURE_COUNT = 2

_MONOCULAR_FEATURE_COUNT = 2
_MONOCULAR_PUPIL_NORM_POS = slice(0, 2)

_BINOCULAR_FEATURE_COUNT = 4
_BINOCULAR_PUPIL_NORM_POS = slice(2, 4)


class Model2D(Model):
    def __init__(self, screen_size=(1, 1), outlier_threshold_pixel=70):
        self.screen_size = screen_size
        self.outlier_threshold_pixel = outlier_threshold_pixel
        self._is_fitted = False
        self._regressor = LinearRegression(fit_intercept=True)

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def set_params(self, **params):
        if params == {}:
            return
        for key, value in params.items():
            setattr(self._regressor, key, np.asarray(value))
        self._is_fitted = True

    def get_params(self):
        has_coef = hasattr(self._regressor, "coef_")
        has_intercept = hasattr(self._regressor, "intercept_")
        if not has_coef or not has_intercept:
            return {}
        return {
            "coef_": self._regressor.coef_.tolist(),
            "intercept_": self._regressor.intercept_.tolist(),
        }

    def fit(self, X, Y, outlier_removal_iterations=1):
        assert X.shape[0] == Y.shape[0], "Required shape: (n_samples, n_features)"
        self._validate_feature_dimensionality(X)
        self._validate_reference_dimensionality(Y)

        if X.shape[0] == 0:
            raise NotEnoughDataError

        polynomial_features = self._polynomial_features(X)
        self._regressor.fit(polynomial_features, Y)

        # iteratively remove outliers and refit the model on a subset of the data
        errors_px, rmse = self._test_pixel_error(X, Y)
        if outlier_removal_iterations > 0:
            filter_mask = errors_px < self.outlier_threshold_pixel
            X_filtered = X[filter_mask]
            Y_filtered = Y[filter_mask]
            n_filtered_out = X.shape[0] - X_filtered.shape[0]

            if n_filtered_out > 0:
                # if we don't filter anything, we can skip refitting the model here
                logger.debug(
                    f"Fitting. RMSE = {rmse:>7.2f}px ..."
                    f" discarding {n_filtered_out}/{X.shape[0]}"
                    f" ({100 * (n_filtered_out) / X.shape[0]:.2f}%)"
                    f" data points as outliers."
                )
                # recursively remove outliers
                return self.fit(
                    X_filtered,
                    Y_filtered,
                    outlier_removal_iterations=outlier_removal_iterations - 1,
                )

        logger.debug(f"Fitting. RMSE = {rmse:>7.2f}px in final iteration.")
        self._is_fitted = True

    def _test_pixel_error(self, X, Y):
        Y_predict = self.predict(X)
        difference_px = (Y_predict - Y) * self.screen_size
        errors_px = np.linalg.norm(difference_px, axis=1)
        root_mean_squared_error_px = np.sqrt(np.mean(np.square(errors_px)))
        return errors_px, root_mean_squared_error_px

    def predict(self, X):
        self._validate_feature_dimensionality(X)
        polynomial_features = self._polynomial_features(X)
        return self._regressor.predict(polynomial_features)

    def _polynomial_features(self, norm_xy):
        # slice data to retain ndim
        norm_x = norm_xy[:, :1]
        norm_y = norm_xy[:, 1:]

        norm_x_squared = norm_x ** 2
        norm_y_squared = norm_y ** 2

        return np.hstack(
            (
                norm_x,
                norm_y,
                norm_x * norm_y,
                norm_x_squared,
                norm_y_squared,
                norm_x_squared * norm_y_squared,
            )
        )

    @staticmethod
    def _validate_reference_dimensionality(Y):
        assert Y.ndim == 2, "Required shape: (n_samples, n_features)"
        assert Y.shape[1] == _REFERENCE_FEATURE_COUNT

    @staticmethod
    @abc.abstractmethod
    def _validate_feature_dimensionality(X):
        raise NotImplementedError


class Model2D_Binocular(Model2D):
    def _polynomial_features(self, norm_xy):
        left = super()._polynomial_features(norm_xy[:, _MONOCULAR_PUPIL_NORM_POS])
        right = super()._polynomial_features(norm_xy[:, _BINOCULAR_PUPIL_NORM_POS])
        return np.hstack((left, right))

    @staticmethod
    def _validate_feature_dimensionality(X):
        assert X.ndim == 2, "Required shape: (n_samples, n_features)"
        assert X.shape[1] == _BINOCULAR_FEATURE_COUNT, (
            f"Received shape: {X.shape}. "
            f"Expected shape (n_samples, {_BINOCULAR_FEATURE_COUNT})"
        )


class Model2D_Monocular(Model2D):
    @staticmethod
    def _validate_feature_dimensionality(X):
        assert X.ndim == 2, "Required shape: (n_samples, n_features)"
        assert X.shape[1] == _MONOCULAR_FEATURE_COUNT, (
            f"Received shape: {X.shape}. "
            f"Expected shape (n_samples, {_MONOCULAR_FEATURE_COUNT})"
        )


class Gazer2D(GazerBase):
    label = "2D"

    def __init__(self, g_pool, *, posthoc_calib=False, calib_data=None, params=None):
        self.posthoc_calib = posthoc_calib
        super().__init__(g_pool, calib_data=calib_data, params=params)

    @classmethod
    def _gazer_description_text(cls) -> str:
        return "2D gaze mapping: use only in controlled conditions; sensitive to movement of the headset (slippage); uses 2d pupil detection result as input."

    def _init_left_model(self) -> Model:
        return Model2D_Monocular(screen_size=self.g_pool.capture.frame_size)

    def _init_right_model(self) -> Model:
        return Model2D_Monocular(screen_size=self.g_pool.capture.frame_size)

    def _init_binocular_model(self) -> Model:
        return Model2D_Binocular(screen_size=self.g_pool.capture.frame_size)

    def _extract_pupil_features(self, pupil_data) -> np.ndarray:
        pupil_features = np.array([p["norm_pos"] for p in pupil_data])
        assert pupil_features.shape == (len(pupil_data), _MONOCULAR_FEATURE_COUNT)
        return pupil_features

    def _extract_reference_features(self, ref_data, monocular=False) -> np.ndarray:
        try:
            if self.g_pool.realtime_ref is None:
                ref_features = np.array([r["norm_pos"] for r in ref_data])
                assert ref_features.shape == (len(ref_data), _REFERENCE_FEATURE_COUNT)
                return ref_features
            else:
                def normalize_point(ref_pt):
                    screen_pos_image_point = ref_pt.reshape(-1, 2)
                    screen_pos_image_point = normalize(
                        screen_pos_image_point[0], self.g_pool.capture.intrinsics.resolution, flip_y=True
                    )
                    return _clamp_norm_point(screen_pos_image_point)

                #unprojected = ref_data[int(len(ref_data)/2)]['screen_pos']
                #projected = self.g_pool.capture.intrinsics.projectPoints(np.array([unprojected]))
                #deprojected_notnorm = self.g_pool.capture.intrinsics.unprojectPoints(projected, normalize=False)
                #deprojected_norm = self.g_pool.capture.intrinsics.unprojectPoints(projected, normalize=True)
                #print("UNPROJECTED:")
                #print(unprojected)
                #print("PROJECTED:")
                #print(projected)
                #print("DEPROJECTED NOTNORM:")
                #print(deprojected_notnorm)
                #print("DEPROJECTED NORM:")
                #print(deprojected_norm)
                #while True:
                #    pass
                projectedPoints = self.g_pool.capture.intrinsics.projectPoints(
                    np.array([r['screen_pos'] for r in ref_data])
                )
                normalized_points = [normalize_point(ref_point) for ref_point in projectedPoints]
                
                ref_features = np.array(normalized_points)
                assert ref_features.shape == (len(ref_data), _REFERENCE_FEATURE_COUNT)
                return ref_features
                #if self.intrinsics is not None:
                #    cyclop_gaze = nearest_intersection_point - cyclop_center
                #    self.last_gaze_distance = np.sqrt(cyclop_gaze.dot(cyclop_gaze))
                #    image_point = self.intrinsics.projectPoints(
                #        np.array([nearest_intersection_point])
                #    )
                #    image_point = image_point.reshape(-1, 2)
                #    image_point = normalize(
                #        image_point[0], self.intrinsics.resolution, flip_y=True
                #    )
                #    image_point = _clamp_norm_point(image_point)
                #    g["norm_pos"] = image_point
        except:
            ref_features = np.array([r["norm_pos"] for r in ref_data])
            assert ref_features.shape == (len(ref_data), _REFERENCE_FEATURE_COUNT)
            return ref_features

    def predict(
        self, matched_pupil_data: T.Iterator[T.List["Pupil"]]
    ) -> T.Iterator["Gaze"]:
        for pupil_match in matched_pupil_data:
            num_matched = len(pupil_match)
            gaze_positions = ...  # Placeholder for gaze_positions

            if num_matched == 2:
                if self.binocular_model.is_fitted and self.right_model.is_fitted and self.left_model.is_fitted:
                    right = self._extract_pupil_features([pupil_match[0]])
                    left = self._extract_pupil_features([pupil_match[1]])
                    X = np.hstack([left, right])
                    assert X.shape[1] == _BINOCULAR_FEATURE_COUNT
                    gaze_positions = self.binocular_model.predict(X).tolist()
                    right_positions = self.right_model.predict(right).tolist()
                    left_positions = self.left_model.predict(left).tolist()
                    topic = "gaze.2d.01."
                else:
                    logger.debug(
                        "Prediction failed because at least one model is not fitted"
                    )
            elif num_matched == 1:
                X = self._extract_pupil_features([pupil_match[0]])
                assert X.shape[1] == _MONOCULAR_FEATURE_COUNT
                if pupil_match[0]["id"] == 0:
                    if self.right_model.is_fitted:
                        gaze_positions = self.right_model.predict(X).tolist()
                        right_positions = self.right_model.predict(X).tolist()
                        left_positions = [[None, None] for i in range(len(gaze_positions))]
                        topic = "gaze.2d.0."
                    else:
                        logger.debug(
                            "Prediction failed because right model is not fitted"
                        )
                elif pupil_match[0]["id"] == 1:
                    if self.left_model.is_fitted:
                        gaze_positions = self.left_model.predict(X).tolist()
                        right_positions = [[None, None] for i in range(len(gaze_positions))]
                        left_positions = self.left_model.predict(X).tolist()
                        topic = "gaze.2d.1."
                    else:
                        logger.debug(
                            "Prediction failed because left model is not fitted"
                        )
            else:
                raise ValueError(
                    f"Unexpected number of matched pupil_data: {num_matched}"
                )

            if gaze_positions is ...:
                continue  # Prediction failed and the reason was logged

            for gaze_pos, right_pos, left_pos in zip(gaze_positions, right_positions, left_positions):
                gaze_datum = {
                    "topic": topic,
                    "norm_pos": gaze_pos,
                    "right_norm_pos": right_pos,
                    "left_norm_pos": left_pos,
                    "confidence": np.mean([p["confidence"] for p in pupil_match]),
                    "timestamp": np.mean([p["timestamp"] for p in pupil_match]),
                    "base_data": pupil_match,
                }
                yield gaze_datum

    def filter_pupil_data(
        self, pupil_data: T.Iterable, confidence_threshold: T.Optional[float] = None
    ) -> T.Iterable:
        pupil_data = list(filter(lambda p: "2d" in p["method"], pupil_data))
        pupil_data = super().filter_pupil_data(pupil_data, confidence_threshold)
        return pupil_data

    def fit_on_calib_data(self, calib_data):
        if not self.posthoc_calib:
            super().fit_on_calib_data(calib_data)
        else:
            #ref_data = calib_data["ref_list"]
            ref_data = self.g_pool.realtime_ref
            if ref_data is None:
                ref_data = calib_data["ref_list"]
            # extract and filter pupil data
            pupil_data = calib_data["pupil_list"]
            pupil_data = self.filter_pupil_data(
                pupil_data, self.g_pool.min_calibration_confidence
            )
            if not pupil_data:
                raise NotEnoughPupilDataError
            if not ref_data:
                raise NotEnoughReferenceDataError
            # match pupil to reference data (left, right, and binocular)
            matches = self.match_pupil_to_ref(pupil_data, ref_data)
            binocular_matches = self.match_pupil_to_ref(pupil_data, ref_data)
            if matches.binocular[0]:
                self._fit_binocular_model(self.binocular_model, binocular_matches.binocular)
                self._fit_monocular_model(self.right_model, matches.right)
                self._fit_monocular_model(self.left_model, matches.left)
            elif matches.right[0]:
                self._fit_monocular_model(self.right_model, matches.right)
            elif matches.left[0]:
                self._fit_monocular_model(self.left_model, matches.left)
            else:
                raise NotEnoughDataError
