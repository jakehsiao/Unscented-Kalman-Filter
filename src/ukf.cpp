#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;
   
  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(7);

  // initial covariance matrix
  P_ = MatrixXd(7, 7);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 30;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 30;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;


  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
  n_aug = 7; // only aug is used this time
  x_ << 0, 0, 0, 0, 0, 0, 0;

  P_.fill(1000);
  P_(5,5) = std_a_ * std_a_;
  P_(6,6) = std_yawdd * std_yawdd; 

  weights = VectorXd(2 * n_aug + 1);


}

UKF::~UKF() {}
MatrixXd UKF::GetSigmaX(){
  // get sigma points of x
  MatrixXd Xsig = MatrixXd(n_aug, 2 * n_aug + 1);
  MatrixXd A = P_.llt().matrixL();
  //set first column of sigma point matrix
  Xsig.col(0)  = x_;

  //set remaining sigma points
  for (int i = 0; i < n_aug; i++)
  {
    Xsig.col(i+1)     = x_ + sqrt(lambda+n_aug) * A.col(i);
    Xsig.col(i+1+n_aug) = x_ - sqrt(lambda+n_aug) * A.col(i);
  }

  return Xsig;

}


MatrixXd UKF::PredSigmaX(MatrixXd &Xsig_aug){
  //predict sigma points
  MatrixXd Xsig_pred = MatrixXd(n_aug, 2 * n_aug + 1);
  for (int i = 0; i< 2*n_aug+1; i++)
  {
    //extract values for better readability
    double p_x = Xsig_aug(0,i);
    double p_y = Xsig_aug(1,i);
    double v = Xsig_aug(2,i);
    double yaw = Xsig_aug(3,i);
    double yawd = Xsig_aug(4,i);
    double nu_a = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if (fabs(yawd) > 0.001) {
        px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
        py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
    }
    else {
        px_p = p_x + v*delta_t*cos(yaw);
        py_p = p_y + v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    //write predicted sigma point into right column
    Xsig_pred(0,i) = px_p;
    Xsig_pred(1,i) = py_p;
    Xsig_pred(2,i) = v_p;
    Xsig_pred(3,i) = yaw_p;
    Xsig_pred(4,i) = yawd_p;
    Xsig_pred(5,i) = nu_a;
    Xsig_pred(6,i) = nu_yawdd;
  }

  return Xsig_pred; 


}

void UKF::PredX(MatrixXd &Xsig_pred){
  // set weights
  double weight_0 = lambda/(lambda+n_aug);
  weights(0) = weight_0;
  for (int i=1; i<2*n_aug+1; i++) {  //2n+1 weights
    double weight = 0.5/(n_aug+lambda);
    weights(i) = weight;
  }

  //predicted state mean
  x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug + 1; i++) {  //iterate over sigma points
    x_ += weights(i) * Xsig_pred.col(i);
  }

  //predicted state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug + 1; i++) {  //iterate over sigma points

    // state difference
    VectorXd x_diff = Xsig_pred.col(i) - x_;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    P_ += weights(i) * x_diff * x_diff.transpose() ;
  }


}

MatrixXd UKF::PredSigmaZ(MatrixXd &Xsig_pred){
  // Get the measurement noise
  if (use_laser_){
    n_z = 2;
    R = MatrixXd(n_z, n_z);
    R << std_laser_px_ * std_laser_px_, 0,
         0, std_laser_py_ * std_laser_py_;

  }

  else if(use_radar_){
    n_z = 3;
    R = MatrixXd(n_z,n_z);
    R <<    std_radr*std_radr, 0, 0,
            0, std_radphi*std_radphi, 0,
            0, 0,std_radrd*std_radrd;
  }


  //transform sigma points into measurement space
  MatrixXd Zsig = MatrixXd(n_z, n_aug);
  

  for (int i = 0; i < 2 * n_aug + 1; i++) {  //2n+1 simga points

    // extract values for better readibility
    double p_x = Xsig_pred(0,i);
    double p_y = Xsig_pred(1,i);
    double v  = Xsig_pred(2,i);
    double yaw = Xsig_pred(3,i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // measurement model
    if (use_radar_){
      Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                        //r
      Zsig(1,i) = atan2(p_y,p_x);                                 //phi
      Zsig(2,i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);   //r_dot
    }
    else if (use_laser_){
      Zsig(0,i) = p_x;
      Zsig(1,i) = p_y;
    }
  }

  return Zsig;

  
}

void UKF::PredZ(MatrixXd &Zsig){
//mean predicted measurement
  z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for (int i=0; i < 2*n_aug+1; i++) {
      z_pred = z_pred + weights(i) * Zsig.col(i);
  }

  //measurement covariance matrix S
  S = MatrixXd(n_z,n_z);
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug + 1; i++) {  //2n+1 simga points
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    S = S + weights(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix
    S = S + R;

  // get NIS
  VectorXd z_diff = z - z_pred;
  double NIS = z_diff.transpose() * S.inverse() * z_diff;
  if (use_radar_){
    NIS_radar_ = NIS;
  }
  else if (use_laser_){
    NIS_laser_ = NIS;
  }
}

void UKF::Update(MatrixXd &Xsig_pred, MatrixXd &Zsig){
    //calculate cross correlation matrix
  MatrixXd Tc = MatrixXd(n_aug, n_z);
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug + 1; i++) {  //2n+1 simga points

    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    // state difference
    VectorXd x_diff = Xsig_pred.col(i) - x_;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    Tc = Tc + weights(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z_diff = z - z_pred;

  //angle normalization
  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();

}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage measurement_pack) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
  
  // first, get the measrement type

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR){
    use_radar_ = true;
    use_laser_ = false;
  }
  else if (measurement_pack.sensor_type == MeasurementPackage::LASER){
    use_radar_ = false;
    use_laser_ = true;
  }



 if (!is_initialized_) {

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      float rho = measurement_pack.raw_measurements_[0];
      float theta = measurement_pack.raw_measurements_[1];
      
      float px = rho * cos(theta);
      float py = rho * sin(theta);

      x_ << px, py, 0, 0, 0, 0, 0;

    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
      x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0, 0, 0, 0;
    }


    previous_timestamp_ = measurement_pack.timestamp_; 
    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }
  
  // get delta t
  delta_t = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0; //not divide so much, 1000000
  previous_timestamp_ = measurement_pack.timestamp_;
  
  // get measurements
  if (use_radar_){
  z << rho, theta, measurement_pack.raw_measurements_[2];
  }
  else if (use_lidar_){
  z << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1];
  }

  MatrixXd Xsig = GetSigmaX();
  MatrixXd Xsig_pred = PredSigmaX(Xsig);
  PredX(Xsig_pred);
  MatrixXd Zsig = PredSigmaZ(Xsig_pred);
  PredZ();
  Update(Xsig_pred, Zsig);

}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
}
