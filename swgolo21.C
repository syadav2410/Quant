///////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                           //   
//                           S  W  G  O  L  O                                                //
//                                                                                           //   
//               Optimization of the footprint of SWGO detector array                        //
//               ----------------------------------------------------                        //
//                                                                                           //   
//  We use a quick and dirty parametrization of muon and electron fluxes as a                //
//  function of radius R for energetic air showers, and a simplified description             //
//  of detector units in terms of efficiency and acceptance, to estimate the                 //
//  uncertainty on gamma fluxes for different energies E, combining them in a                //
//  utility function as a function of detector unit spacing D.                               //
//  The crude model is only meant to be a demonstration of the structure of the              //
//  optimization task.                                                                       //
//                                                                                           //
//  This version of the code fits both the position of showers (x0,y0) and their             //
//  polar and azimuthal angles through a likelihood maximization. The fit is performed       //
//  twice - once for the gamma and once for the proton hypothesis. The two values of logL    //
//  at maximum are used in the construction of a likelihood-ratio test statistic.            //
//  The distribution of this TS for the two hypotheses is the basis of the extraction        //
//  of the uncertainty on the signal fraction in the shower batches, separately for each     //
//  generated energy E.                                                                      //
//                                                                                           //
//                                                                                           //       
//                                                       T. Dorigo, December 2022            //
//                                                                                           //
///////////////////////////////////////////////////////////////////////////////////////////////

#include "TH2.h"
#include "TH1.h"
#include "TProfile.h"
#include "TStyle.h"
#include "TCanvas.h" 
#include "TROOT.h"
#include "TMath.h"
#include <math.h>
#include "TRandom.h"
#include "TRandom3.h"
#include "Riostream.h"

#include <stdio.h>
#include <iostream>
#include <sstream>
#include <fstream>

using namespace std;

// Constants and control settings
// ------------------------------ 
static const double epsilon        = 0.00000001;
static const double largenumber    = 10000000000000.;
static const double pi             = 3.1415926;
static const double sqrt2pi        = sqrt(2.*pi);
static const double gamma_EM       = 0.5772156649; // Euler-Mascheroni constant
static const bool   debug          = false; // if on, lots of printouts
static const bool   plotdistribs   = false; // if on, plot densities per m^2
static const bool   speedup        = false; // if on, we do fewer calculations for x0,y0, sigma2
static const bool   usetrueXY      = false; // if on, avoids fitting for x0,y0 of shower // NB don't have all true!
static const bool   usetrueAngs    = false; // if on, avoids fitting for theta, phi of shower
static const bool   fixShowerPos   = false; // if on, showers are always generated at the same locations 
static const bool   OrthoShowers   = false; // if on, showers come down orthogonally to the ground
static const bool   SlantedShowers = false; // if on, showers come in at pi/4
static const bool   hexaShowers    = true;  // used if fixShowerPos is true to set the geometry of showers
static const bool   addSysts       = false; // if on, we mess up a bit the distributions to avoid trouble in JSD calculations
static const bool   checkUtility   = false; // if on, the utility is recomputed after a detector move
static const bool   readGeom       = false; // if on, reads detector positions from file
static const bool   writeGeom      = true;  // if on, writes final detector positions to file
static const int    maxUnits       = 10000; // max number of detectors deployed
static const int    maxEvents      = 10000; // max events simulated for templates
static const int    maxEpochs      = 5000;  // max number of epochs of utility maximization
static const int    maxEbins       = 50;    // max considered number of energy points
static const int    maxPairs       = 100;   // max pair of bins for fg estimate (not used anymore)
static const int    maxRbins       = 1000;  // max number of bins in R where to average utility (if commonMode=1)

// Global parameters required for dimensioning arrays, and defaults
// ----------------------------------------------------------------
static int Nunits     = 500;
static int Nbatch     = 500;
static int Nevents    = 500;
static int NEvalues   = 5;
static int Nepochs    = 100;
static int NRbins     = 100;   // Number of R bins in xy plane, to average derivatives of U, for commonMode=1

// Other parameters defining run
// -----------------------------
static int Ncycles       = 200;    // Number of SGD cycles of learning rate decrease
static int Nsteps        = 30.;    // Steps in likelihood maximization - NB, 50 is safe but slow
static double spanR      = 0.;     // This gets defined based on the initial layout of x[], y[]
static double Xoffset    = 0.;     // Used to study behaviour of maximization and "drift to interesting region"
static double Yoffset    = 0.;     // Same as above
static double XDoffset   = 0.;     // With these we put detector units offset instead of showers
static double YDoffset   = 0.;     // Same as above
static double Rmin       = 1.;     // Min distance of shower center from detector center (avoiding divergence of gradients) 
static double Rslack     = 0.;     // This is to generate showers around detectors and not only over them
static int commonMode    = 0;      // choice to vary all xy (0), R (1), or common center (2) of array
static double posrate    = 1.;     // Multiplier of step for x0,y0 likelihood maximization
static double posrateang = 1.;     // same, for angle determination
static double MaxUtility = 50.;    // max value of plotted utility in U graph
static double TankArea   = pi*4.;  // 2 m radius tanks 

// New random number generator
// ---------------------------
static TRandom3 * myRNG = new TRandom3();

// Shower parameters
// -----------------
static double param_mufromp[4][maxEbins];
static double param_efromp[4][maxEbins];
static double param_mufromg[4][maxEbins];
static double param_efromg[4][maxEbins];
static double Fg[maxEbins];
static double LLRmin[maxEbins];
static double mug[maxUnits]; // We need the exp values for an event in each unit, to 
static double eg[maxUnits];  // compute the uncertainty on the LLR for the propagation
static double mup[maxUnits]; // of dU/dx, dU/dy
static double ep[maxUnits];
static double Utility;
static double UtilityErr;
static bool isGamma;
static double JS;

// Detector positions and parameters
// ---------------------------------
static double x[maxUnits];
static double y[maxUnits];
static double xprev[maxUnits];
static double yprev[maxUnits];
static double TrueX0[maxEvents];
static double TrueY0[maxEvents];
static double TrueTheta[maxEvents];
static double TruePhi[maxEvents];
static int shape = 3; // 0 = hexagonal 1 = taxi 2 = spiral 3 = circular - this gets reassigned on start
static double spiral_reduction = 0.999; // for spiral layout
static double step_increase    = 1.02;  // for spiral layout
static double dldrm[maxUnits][maxEvents]; // this is used to store values during calculations of derivatives, to avoid multiple calculations

// Number of mus and es in each detector unit
// ------------------------------------------
static int Nmu[maxUnits];
static int Ne[maxUnits];

// Measured values of position of shower
// -------------------------------------
static double x0meas[maxEbins][maxEvents][2];
static double y0meas[maxEbins][maxEvents][2];
static double thmeas[maxEbins][maxEvents][2];
static double phmeas[maxEbins][maxEvents][2];

// Test statistic discriminating gamma from proton showers, for current batch
// --------------------------------------------------------------------------
static double logLRT[maxEbins][maxEvents];  // for templates
static bool   IsGamma[maxEbins][maxEvents]; // for templates
static double sigma2LRT[maxEbins][maxEvents]; // for derivative calculation
static double GammaFraction[maxEbins];
static double GammaFracErr[maxEbins];
static double TrueGammaFraction[maxEbins] = {0.5,0.5,0.5,0.5,0.5}; // {0.1, 0.2, 0.3, 0.4, 0.5}; 
static int Nrep4sigma = 5;

// Weight of different energy points in utility
// --------------------------------------------
static double weight[maxEbins];

// Static histos
// -------------
static double maxdxy = 1000.;
static TH1D * DXP = new TH1D ("DXP", "", 200, -maxdxy, maxdxy);
static TH1D * DYP = new TH1D ("DYP", "", 200, -maxdxy, maxdxy);
static TH1D * DXG = new TH1D ("DXG", "", 200, -maxdxy, maxdxy);
static TH1D * DYG = new TH1D ("DYG", "", 200, -maxdxy, maxdxy);
static TH1D * DTHG = new TH1D ("DTHG", "", 200, -pi/2., pi/2.);
static TH1D * DPHG = new TH1D ("DPHG", "", 200, -2*pi, 2*pi);
static TH1D * DTHP = new TH1D ("DTHP", "", 200, -pi/2., pi/2.);
static TH1D * DPHP = new TH1D ("DPHP", "", 200, -2*pi, 2*pi);
static TH2D * DTHPvsT = new TH2D ("DTHPvsT", "", 50, -pi/2.,pi/2.,50, 0.,pi/2. );
static TH2D * DTHGvsT = new TH2D ("DTHGvsT", "", 50, -pi/2.,pi/2.,50, 0.,pi/2. );

static TH1D * DX0g = new TH1D("DX0g", "", 500, -100., 100.);
static TH1D * DY0g = new TH1D("DY0g", "", 500, -100., 100.);
static TH1D * DThg = new TH1D("DThg", "", 500, -pi/2., pi/2.);
static TH1D * DPhg = new TH1D("DPhg", "", 500, 0., 2.*pi);
static TH1D * DX0p = new TH1D("DX0p", "", 500, -100., 100.);
static TH1D * DY0p = new TH1D("DY0p", "", 500, -100., 100.);
static TH1D * DThp = new TH1D("DThp", "", 500, -pi/2., pi/2.);
static TH1D * DPhp = new TH1D("DPhp", "", 500, 0., 2.*pi);

static TH1D * SigLRT = new TH1D ("SigLRT","", 100, 0., 40000.);
static TH2D * SigLvsDR= new TH2D ("SigLvsDR","",100, 0., 40000., 100, 0., 400.);
static TH2D * DL = new TH2D ("DL","",100,0.5, 100.5,100,-50000.,+50000.);
static TH2D * StepsizeX = new TH2D ("StepsizeX","",100, 0.5, 100.5,100,0.,50.);
static TH2D * StepsizeY = new TH2D ("StepsizeY","",100, 0.5, 100.5,100,0.,50.);
static TH2D * StepsizeT = new TH2D ("StepsizeT","",100, 0.5, 100.5,100,0.,2.);
static TH2D * StepsizeP = new TH2D ("StepsizeP","",100, 0.5, 100.5,100,0.,2.);


// Factorial function
// ------------------
long double Factorial (int n) {
  if (n==0)  return 1.;
  if (n>170) return 10E300;
  return Factorial(n-1)*n;
}

// Learning rate scheduler
// -----------------------
double LR_Scheduler (double LR0, int epoch) {
    double par[3] = {-0.04,0.1,0.2};
    double x = 100.*epoch/Nepochs;
    return LR0*exp(par[0]*x)*(par[1]+pow(cos(par[2]*x),2));
}

// Function computing the effective distance of a detector from shower center, given a slanted shower.
// To compute the distance to the shower axis, we have to project along theta (via a cos(theta) factor in 
// the longitudinal projection)the distance computed in the "rotated" space. We define:
// -  u = (x-x0)*sin(phi)-(y-y0)*cos(phi)
// -  v = (x-x0)*cos(phi)+(y-y0)*sin(phi)
// and R = sqrt(u*u+v*v*cos^2 theta)
// ------------------------------------------------------------------------------------------------------
double EffectiveDistance (double xd, double yd, double x0, double y0, double theta, double phi, int mode) {
    double dx = xd-x0;
    double dy = yd-y0;
    double r = sqrt(pow(dx*sin(phi)-dy*cos(phi),2)+pow((dx*cos(phi)+dy*sin(phi))*cos(theta),2));
    if (mode==0) { // return distance
        return r; 
    } else if (mode==1) { // return derivative wrt xd
        return (sin(phi)*(dx*sin(phi)-dy*cos(phi))  + pow(cos(theta),2)*cos(phi)*(dx*cos(phi)+dy*sin(phi)))/r; 
    } else if (mode==2) { // return derivative wrt yd
        return (-cos(phi)*(dx*sin(phi)-dy*cos(phi)) + pow(cos(theta),2)*sin(phi)*(dx*cos(phi)+dy*sin(phi)))/r;
    } else if (mode==3) { // return derivative wrt theta
        return -sin(theta)*cos(theta)*pow(dx*cos(phi)+dy*sin(phi),2)/r;
    } else {
        return r;
    }
}

// Function parametrizing muon content in gamma showers
// ----------------------------------------------------
double MuFromG (int ie, double R, int mode) {
    if (mode==0) { // return function value
        double flux = param_mufromg[0][ie] / (pow(R,param_mufromg[1][ie])+param_mufromg[2][ie]);
        return TankArea * flux;
    } else if (mode==1) { // return derivative with respect to R
        return -TankArea * param_mufromg[0][ie]*param_mufromg[1][ie] * pow(R,param_mufromg[1][ie]-1.) /
               pow(pow(R,param_mufromg[1][ie])+param_mufromg[2][ie],2); 
    } else {
        return 0.;
    }
}
// Function parametrizing bgr content in gamma showers
// ---------------------------------------------------
double EFromG (int ie, double R, int mode) {
    if (mode==0) { // return function value
        double flux = param_efromg[0][ie] / (pow(R,param_efromg[1][ie])+param_efromg[2][ie]);
        return TankArea * flux;
    } else if (mode==1) { // return derivative with respect to R
        return -TankArea * param_efromg[0][ie]*param_efromg[1][ie] * pow(R,param_efromg[1][ie]-1.) /
               pow(pow(R,param_efromg[1][ie])+param_efromg[2][ie],2); 
    } else {    
        return 0.;
    }
}

// Function parametrizing muon content in proton showers
// ---------------------------------------------.-------
double MuFromP (int ie, double R, int mode) {
    if (mode==0) { // Return function value
        double flux = param_mufromp[0][ie] / (pow(R,param_mufromp[1][ie])+param_mufromp[2][ie]);
        return TankArea * flux;
    } else if (mode==1) { // return derivative with respect to R
        return -TankArea * param_mufromp[0][ie]*param_mufromp[1][ie] * pow(R,param_mufromp[1][ie]-1.) /
               pow(pow(R,param_mufromp[1][ie])+param_mufromp[2][ie],2); 
    } else {    
        return 0.;
    }
}

// Function parametrizing bgr content in proton showers
// ----------------------------------------------------
double EFromP (int ie, double R, int mode) {
    if (mode==0) { // return function value
        double flux = param_efromp[0][ie] / (pow(R,param_efromp[1][ie])+param_efromp[2][ie]);
        return TankArea * flux;
    } else if (mode==1) { // return derivative with respect to R
        return -TankArea * param_efromp[0][ie]*param_efromp[1][ie] * pow(R,param_efromp[1][ie]-1.) /
               pow(pow(R,param_efromp[1][ie])+param_efromp[2][ie],2); 
    } else { 
        return 0.;
    }
}

// This function defines a layout which draws "MODE" on the ground
// ---------------------------------------------------------------
void DrawMODE(int idfirst, double x_lr, double y_lr, double xystep, double xstep, double ystep) {

    Nunits += 132;
    // M:
    for (int id=0; id<10; id++) {
        x[idfirst+id] = XDoffset + x_lr;
        y[idfirst+id] = YDoffset + y_lr+id*xystep;
    }
    for (int id=0; id<7; id++) {
        x[idfirst+id+10] = XDoffset + x_lr+id*xstep;
        y[idfirst+id+10] = YDoffset + y_lr+10*xystep-id*ystep;
    }
    for (int id=0; id<7; id++) {
        x[idfirst+id+17] = XDoffset + x_lr+7*xstep+id*xstep;
        y[idfirst+id+17] = YDoffset + y_lr+10*xystep-7*ystep+id*ystep;
    }
    for (int id=0; id<11; id++) {
        x[idfirst+id+24] = XDoffset + x_lr+14*xstep;
        y[idfirst+id+24] = YDoffset + y_lr+10*xystep-id*xystep;
    }
    // O:
    for (int id=0; id<10; id++) {
        x[idfirst+id+35] = XDoffset + x_lr+14*xstep+3*xystep;
        y[idfirst+id+35] = YDoffset + y_lr+id*xystep;
    }
    for (int id=0; id<7; id++) {
        x[idfirst+id+45] = XDoffset + x_lr+14*xstep+3*xystep+id*xystep;
        y[idfirst+id+45] = YDoffset + y_lr+10*xystep;
    }
    for (int id=0; id<10; id++) {
        x[idfirst+id+52] = XDoffset + x_lr+14*xstep+10*xystep;
        y[idfirst+id+52] = YDoffset + y_lr+10*xystep-id*xystep;
    }
    for (int id=0; id<7; id++) {
        x[idfirst+id+62] = XDoffset + x_lr+14*xstep+10*xystep-id*xystep;
        y[idfirst+id+62] = YDoffset + y_lr;
    }
    // D:
    for (int id=0; id<10; id++) {
        x[idfirst+id+69] = XDoffset + x_lr+14*xstep+13*xystep;
        y[idfirst+id+69] = YDoffset + y_lr+id*xystep;
    }
    for (int id=0; id<2; id++) {
        x[idfirst+id+79] = XDoffset + x_lr+14*xstep+13*xystep+id*xystep;
        y[idfirst+id+79] = YDoffset + y_lr+10*xystep;
    }
    for (int id=0; id<6; id++) {
        x[idfirst+id+81] = XDoffset + x_lr+14*xstep+15*xystep+id*xstep;
        y[idfirst+id+81] = YDoffset + y_lr+10*xystep-id*ystep;
    }
    for (int id=0; id<3; id++) {
        x[idfirst+id+87] = XDoffset + x_lr+19*xstep+15*xystep;
        y[idfirst+id+87] = YDoffset + y_lr+10*xystep-5*ystep-id*xystep;
    }
    for (int id=0; id<6; id++) {
        x[idfirst+id+90] = XDoffset + x_lr+19*xstep+15*xystep-id*xstep;
        y[idfirst+id+90] = YDoffset + y_lr+7*xystep-5*ystep-id*ystep;
    }
    for (int id=0; id<2; id++) {
        x[idfirst+id+96] = XDoffset + x_lr+13*xstep+15*xystep-id*xystep;
        y[idfirst+id+96] = YDoffset + y_lr;
    }
    // E:
    for (int id=0; id<10; id++) {
        x[idfirst+id+98] = XDoffset + x_lr+19*xstep+18*xystep;
        y[idfirst+id+98] = YDoffset + y_lr+id*xystep;
    }
    for (int id=0; id<8; id++) {
        x[idfirst+id+108] = XDoffset + x_lr+19*xstep+18*xystep+id*xystep;
        y[idfirst+id+108] = YDoffset + y_lr;
        x[idfirst+id+116] = XDoffset + x_lr+19*xstep+18*xystep+id*xystep;
        y[idfirst+id+116] = YDoffset + y_lr+5*xystep;
        x[idfirst+id+124] = XDoffset + x_lr+19*xstep+18*xystep+id*xystep;
        y[idfirst+id+124] = YDoffset + y_lr+10*xystep;
    }
}


// Define the current geometry by updating detector positions
// ----------------------------------------------------------
void DefineLayout (double detSpacing, double SpacingStep) {

    // We create a grid of detector positions.
    // We pave the xy space with a spiral from (0,0):
    // one step up, one right, two down, two left, three up, 
    // three right, four down, four left, etcetera.
    // The parameter SpacingStep widens the step progressively
    // from d to larger values. This allows to study layouts
    // with different density variations from center to periphery;
    // a future extension with more parameters will probe more
    // complex configurations by changing the SpacingStep with
    // a functional form.
    // - d = spacing at start of spiral
    // - SpacingStep = rate of increase of spacing
    // - shape = shape of layout (0 = hexagonal, 1 = taxi, 2 = spiral, 3 = circles)
    // ----------------------------------------------------------------------------
    x[0] = XDoffset;
    y[0] = YDoffset;
    int id = 1;
    if (shape==0) { // hexagonal grid
        double deltau = detSpacing;
        double deltav = detSpacing;
        double deltaz = detSpacing;
        int nstepsu = 1;
        int nstepsv = 1;
        int nstepsz = 1;
        double cos30 = sqrt(3.)/2.;
        double sin30 = 0.5;
        int parity = 1.;
        do {
            for (int is=0; is<nstepsu && id<Nunits; is++) {
                x[id] = x[id-1];
                y[id] = y[id-1] + deltau;
                id++;
            }
            deltau = -deltau;
            for (int is=0; is<nstepsv && id<Nunits; is++) {
                x[id] = x[id-1] + deltav*cos30;
                y[id] = y[id-1] + deltav*sin30;
                id++;
            }
            deltav = -deltav;
            for (int is=0; is<nstepsz && id<Nunits; is++) {
                x[id] = x[id-1] + deltaz*cos30;
                y[id] = y[id-1] - deltaz*sin30;
                id++;
            }
            deltaz = -deltaz;
            if (parity==-1) {
                nstepsv++;
            } else {
                nstepsu++;
                nstepsv++;
                nstepsz++;
            }
            parity *= -1;

            // After half cycle we increase the steps size
            // -------------------------------------------
            if (deltau>0) {
                deltau = deltau + SpacingStep;
            } else {
                deltau = deltau - SpacingStep;
            }
            if (deltav>0) {
                deltav = deltav + SpacingStep;
            } else {
                deltav = deltav - SpacingStep;
            }
            if (deltaz>0) {
                deltaz = deltaz + SpacingStep;
            } else {
                deltaz = deltaz - SpacingStep;
            }

        } while (id<Nunits); 
    } else if (shape==1) { // square grid
        int n_steps = 1;
        double deltax = detSpacing;
        double deltay = detSpacing;
        do {
            for (int is=0; is<n_steps && id<Nunits; is++) {
                x[id] = x[id-1];
                y[id] = y[id-1] + deltay;
                if (debug) cout << "id = " << id << " x,y = " << x[id] << "," <<  y[id] << endl;
                id++;
            }
            deltay = -deltay;
            for (int is=0; is<n_steps && id<Nunits; is++) {
                x[id] = x[id-1] + deltax;
                y[id] = y[id-1];
                if (debug) cout << "id = " << id << " x,y = " << x[id] << "," <<  y[id] << endl;
                id++;
            }
            deltax = -deltax;
            n_steps++;
            if (deltax>0) {
                deltax = deltax + SpacingStep;
            } else {
                deltax = deltax - SpacingStep;
            }
            if (deltay>0) {
                deltay = deltay + SpacingStep;
            } else {
                deltay = deltay - SpacingStep;
            }
        } while (id<Nunits); 
    } else if (shape==2) { // smooth spiral
        double delta = detSpacing;
        double angle0 = 1.; // better not be a submultiple of 2*pi if spiral_red is close to 1
        double angle = angle0;
        do {
            x[id] = x[id-1] + delta*cos(angle);
            y[id] = y[id-1] + delta*sin(angle);
            id++;
            angle0 = angle0*spiral_reduction;
            angle += angle0;
            delta = delta*SpacingStep; // step_increase;
            if (debug) cout << id << " " << angle << " " << delta << " " << cos(angle) << " " << sin(angle) << " " << x[id] << " " << y[id] << endl;
        } while (id<Nunits); 
    } else if (shape==3) {
        double r = detSpacing;
        do {
            double n = 6.*r/detSpacing;
            for (int ith=0; ith<n && id<Nunits; ith++) {
                double theta = ith*2*pi/n;
                x[id] = XDoffset + r*cos(theta);
                y[id] = YDoffset + r*sin(theta);
                id++;
            }
            r = r + SpacingStep;
        } while (id<Nunits);
    } else if (shape==4) { // Random 2D box distribution
        for (int id=0; id<Nunits; id++) {
            double halfspan = detSpacing*sqrt(Nunits);
            x[id] = XDoffset + myRNG->Uniform(-halfspan,halfspan);
            y[id] = YDoffset + myRNG->Uniform(-halfspan,halfspan);
        }
    } else if (shape==5) { // Word layout
        int idfirst   = 0;
        double x_lr   = -410;
        double y_lr   = -50;
        double xystep = 10;
        double xstep  = 7;
        double ystep = 7;
        Nunits = 0;
        DrawMODE(idfirst,x_lr,y_lr,xystep,xstep,ystep);
        idfirst   = 132;
        x_lr   = 40;
        y_lr   = -50;
        xystep = 10;
        xstep  = 7;
        ystep = 7;
        DrawMODE(idfirst,x_lr,y_lr,xystep,xstep,ystep);
        //    Nunits = 132;
        idfirst = 264;
        x_lr = -200;
        y_lr = 250;
        xystep = 10;
        xstep = 7;
        ystep = 7;
        DrawMODE(idfirst,x_lr,y_lr,xystep,xstep,ystep);
        idfirst = 396;
        // Nunits = 396;
        x_lr = -200;
        y_lr = -350;
        xystep = 10;
        xstep = 7;
        ystep = 7;
        DrawMODE(idfirst,x_lr,y_lr,xystep,xstep,ystep);
        Nunits = 528;
    } else if (shape==6) { // rectangle
        for (int id=0; id<Nunits; id++) {
            x[id] = XDoffset -detSpacing/2.+detSpacing*(id%2);
            y[id] = YDoffset -Nunits*0.25*detSpacing+detSpacing*(id/2);
        }
    } // end if shape

    // Define span x and y of generated showers to illuminate initial layout
    // ---------------------------------------------------------------------
    for (int id=0; id<Nunits; id++) {
        double thisr = sqrt(x[id]*x[id]+y[id]*y[id]);
        if (thisr>spanR) spanR = thisr;
    }

    return;
}

// Function that saves the layout data to file
// -------------------------------------------
void SaveLayout () {
    string detPath  = "./MODE/dets/";
    ofstream detfile;
    std::stringstream sstr;
    char num[40];
    sprintf (num,"Nsh=%d_Nu=%d_Nep=%d_Layout%d", Nbatch, Nunits, Nepochs, shape);
    sstr << "Layout_";
    string detPositions = detPath  + sstr.str() + num + ".txt";
    detfile.open(detPositions);
    for (int id=0; id<Nunits; id++) {
        detfile << x[id] << " " << y[id] << " " << endl;
    }
    detfile.close();
}
    
// Function that reads the layout data from file
// ---------------------------------------------
void ReadLayout () {
    string detPath  = "./MODE/dets/";
    ifstream detfile;
    std::stringstream sstr;
    char num[40];
    sprintf (num,"Nsh=%d_Nu=%d_Nep=%d_Layout%d", Nbatch,Nunits,Nepochs,shape);
    sstr << "Layout_";
    string detPositions = detPath  + sstr.str() + num + ".txt";
    detfile.open(detPositions);
    double e;
    for (int id=0; id<Nunits; id++) {
        detfile >> e;
        x[id] = e;
        detfile >> e;
        y[id] = e;
        cout << "   Unit # " << id << ": x,y = " << x[id] << " " << y[id] << endl;
    }
    detfile.close();

    // Define span x and y of generated showers to illuminate initial layout
    // ---------------------------------------------------------------------
    for (int id=0; id<Nunits; id++) {
        double thisr = sqrt(x[id]*x[id]+y[id]*y[id]);
        if (thisr>spanR) spanR = thisr;
    }

}

// Generate shower and distribute particle signals in units
// --------------------------------------------------------
void GenerateShower (int ie, int is, bool isGamma) {
 
    // Define polar and azimuthal angle of shower
    // ------------------------------------------
    if (OrthoShowers) {
        TrueTheta[is] = 0.;
        TruePhi[is]   = 0.;
    } else if (SlantedShowers) {
        TrueTheta[is] = pi/4.;
        TruePhi[is]   = myRNG->Uniform(pi);
    } else {
        do {
            TrueTheta[is] = fabs(myRNG->Gaus()*pi/8.); 
        } while (TrueTheta[is]>=pi/2.-pi/8.-epsilon);
        TruePhi[is] = myRNG->Uniform(pi);
    }

    // We get the number density per m^2 of muons and other particles
    // as a function of R at the nominal detector position, for all detector units
    // Note, this matrix does not depend on energy - it is regenerated
    // for every energy point (we only use it inside the ie loop in
    // the code calling this function)
    // ---------------------------------------------------------------------------
    for (int id=0; id<Nunits; id++) {   

        Nmu[id] = 0;
        Ne[id]  = 0;

        double R = EffectiveDistance(x[id],y[id],TrueX0[is],TrueY0[is],TrueTheta[is],TruePhi[is],0);
        if (R<Rmin) R = Rmin; // For consistency with this minimum value elsewhere

        // Irrespective of whether we have to generate Nmu, Ne for gamma or proton
        // here we compute expected fluxes for both, as this information is useful
        // later on (to get sigma2 in findlogLR later, e.g.)
        // -----------------------------------------------------------------------
        double ct = cos(TrueTheta[is]);
        mug[id] = MuFromG(ie,R,0)*ct;
        eg[id]  = EFromG(ie,R,0)*ct;
        mup[id] = MuFromP(ie,R,0)*ct;
        ep[id]  = EFromP(ie,R,0)*ct;
        if (isGamma) {
            if (mug[id]>0.) Nmu[id] = myRNG->Poisson(mug[id]); // otherwise it remains zero
            if (eg[id]>0.)  Ne[id]  = myRNG->Poisson(eg[id]);  // otherwise it remains zero
        } else {
            if (mup[id]>0.) Nmu[id] = myRNG->Poisson(mup[id]); // otherwise it remains zero
            if (ep[id]>0.)  Ne[id]  = myRNG->Poisson(ep[id]);  // otherwise it remains zero
        }
    } // end id loop
    return;
}

// Find most likely position of shower center by max likelihood
// NB even in Asimov approximation, we rely on the definition of
// Nmu, Ne as otherwise the resampling we do to compute sigma2
// would not work. See note below (in routine FindLogLR)
// -------------------------------------------------------------
double FindShowerPos(int ie, int is, bool isGamma) {

    // Initialize shower position at max flux
    // --------------------------------------
    double currentX0 = 0.;
    double currentY0 = 0.;
    if (usetrueXY) {
        currentX0 = TrueX0[is];
        currentY0 = TrueY0[is];
    } else {
        double xmed  = 0.;
        double ymed  = 0.; 
        //double xmed2 = 0.;
        //double ymed2 = 0.;
        double sum   = 0.;
        for (int id=0; id<Nunits; id++) {
            double thisFlux = Nmu[id] + Ne[id];
            double xterm = thisFlux*x[id];
            double yterm = thisFlux*y[id];
            xmed += xterm;
            ymed += yterm;
            //xmed2+= xterm*xterm;
            //ymed2+= yterm*yterm;
            sum  += thisFlux;
        }
        if (sum>0.) {
            currentX0 = xmed/sum;
            currentY0 = ymed/sum;
            //xmed2 = xmed2/sum - pow(currentX0,2);
            //ymed2 = ymed2/sum - pow(currentY0,2);
        }
    }

    // Initialize shower angles
    // ------------------------
    double currentPhi   = 0.;
    double currentTheta = 0.;
    if (usetrueAngs) {
        currentTheta = TrueTheta[is];
        currentPhi   = TruePhi[is];
    } else {

        // We initialize phi and theta by finding the principal axis of the particle distribution on the
        // ground, which provides phi, and the ratio of square of variances along phi+pi and phi, which
        // provides cos(theta).
        // Note that muons and electrons here (and elsewhere) provide the same information - which needs
        // not be the case if they have different radial distributions. This is something to improve.
        // ---------------------------------------------------------------------------------------------
        double minsum = largenumber;
        for (int iphi=0; iphi<180; iphi+=1) { // we go by 5 degree steps
            double phi = iphi*pi/180.;
            double sum = 0.;
            for (int id=0; id<Nunits; id++) {
                if (Nmu[id]+Ne[id]==0) continue;
                double rdet2 = pow(x[id]-currentX0,2)+pow(y[id]-currentY0,2);
                double phid = pi/2.;
                if (x[id]-currentX0!=0.) phid = atan((y[id]-currentY0)/(x[id]-currentX0));
                sum += rdet2*pow(sin(phi-phid),2)*(Nmu[id]+Ne[id]);
            }
            if (sum<minsum) {
                minsum = sum;
                currentPhi = phi;
            }
        }
        // OK now get theta as acos(sqrt(var_orth/var_phi));
        // -------------------------------------------------
        double varphi = 0.;
        double varort = 0.;
        for (int id=0; id<Nunits; id++) {
            if (Nmu[id]+Ne[id]==0) continue;
            double rdet2 = pow(x[id]-currentX0,2)+pow(y[id]-currentY0,2);
            double phid = pi/2.;
            if (x[id]-currentX0!=0.) phid = atan((y[id]-currentY0)/(x[id]-currentX0));
            varphi += rdet2*pow(cos(currentPhi-phid),2)*(Nmu[id]+Ne[id]);
            varort += rdet2*pow(sin(currentPhi-phid),2)*(Nmu[id]+Ne[id]);
        }
        varphi = varphi/(Nunits-1);
        varort = varort/(Nunits-1);
        if (varphi*varort>0.) currentTheta = acos(sqrt(varort/varphi));
        if (debug) cout << "varort, varphi, minsum, phi, acos = " << varort << " " << varphi << " " << minsum 
                        << " " << currentPhi << " " << acos(sqrt(varort/varphi)) << endl;

        if (debug) cout << " True theta,phi = " << TrueTheta[is] << " " << TruePhi[is] 
                        << " Meas theta,phi = " << currentTheta << " " << currentPhi << endl;

    }

    // Define and zero variables used in loop below
    // --------------------------------------------
    double logL          = 0.;
    double prevlogL      = 0.;
    double dlogLdX0;
    double dlogLdY0;
    double dlogLdTh;
    double dlogLdPh;
    double prevdlogLdX0  = 0.;
    double prevdlogLdY0  = 0.;
    double prevdlogLdTh  = 0.;
    double prevdlogLdPh  = 0.;
    double LearningRateX = 0.000001; 
    double LearningRateY = 0.000001;
    double LearningRateTh= 0.0001;
    double LearningRatePh= 0.001;
    int istep = 0;

    // Loop to maximize logL and find X0, Y0, Theta, Phi of shower
    // -----------------------------------------------------------
    do {
        prevlogL = logL;
        logL     = 0.;
        dlogLdX0 = 0.;
        dlogLdY0 = 0.;
        dlogLdTh = 0.;
        dlogLdPh = 0.;
        double sp  = sin(currentPhi);
        double cp  = cos(currentPhi);
        double st  = sin(currentTheta);
        double ct  = cos(currentTheta);
        double c2t = pow(ct,2);

        // Sum contributions from all detectors to logL and derivatives
        // ------------------------------------------------------------
        for (int id=0; id<Nunits; id++) {

            double dx  = x[id]-currentX0;
            double dy  = y[id]-currentY0;

            // How far is this unit from assumed shower center, projected along direction?
            // ---------------------------------------------------------------------------
            double thisR = EffectiveDistance(x[id],y[id],currentX0,currentY0,currentTheta,currentPhi,0);
            if (thisR<Rmin) thisR = Rmin;
            // if (debug) cout << "  in logLR x,y= " << x[id] << " " << y[id] << " currxy= " << currentX0 << " " << currentY0 << " thisR = " << thisR << endl;            

            // Note we assume we know the shower energy ("ie" is fixed)
            // --------------------------------------------------------
            double lambdaMu0, lambdaE0;
            if (isGamma) { // gamma HYPOTHESIS
                lambdaMu0 = MuFromG(ie, thisR, 0);
                lambdaE0  = EFromG(ie, thisR, 0);
            } else {        // proton HYPOTHESIS        
                lambdaMu0 = MuFromP(ie, thisR, 0);
                lambdaE0  = EFromP(ie, thisR, 0);
            }
            if (lambdaMu0!=lambdaMu0 || lambdaE0!=lambdaE0) cout << "lambda0 problem " << ie << " " << thisR << " " << currentX0 << " " << currentY0 << " " << currentTheta << " " << currentPhi << endl;

            double lambdaMu = lambdaMu0*ct; // For tilted showers the flux is reduced as the cross section of the tank is
            double lambdaE  = lambdaE0*ct;  // (here we are not modeling the full cylinder, which would modify this)
            if (lambdaMu*lambdaE==0.) { // zero total flux predictions ?
                cout << "     Zero flux for this config and shower" << endl;
                return -largenumber;
            }
            // if (debug) cout << "  id= " << id << "lambdas = " << lambdaMu  << " " << lambdaE << endl;

            // logL is sum of -lambda + N log(lambda) - log(N!):
            // -------------------------------------------------
            logL -= lambdaMu;
            logL -= lambdaE;
            logL += Nmu[id] * log(lambdaMu);
            logL += Ne[id]  * log(lambdaE);

            if (logL!=logL) {
                cout << "  logL = " << logL << " done with " << -lambdaMu << " " << -lambdaE << " " << Nmu[id] << " " << Ne[id] 
                     << -Nmu[id]*log(lambdaMu) << " " << -Ne[id]*log(lambdaE) << endl;
                SaveLayout();
                return 0.;
            }

            double dRdX0 = (sp*(-dx*sp+dy*cp)-c2t*cp*(dx*cp+dy*sp))/thisR;
            double dRdY0 = (-cp*(-dx*sp+dy*cp)-c2t*sp*(dx*cp+dy*sp))/thisR;
            double dRdTh = (-st*ct*pow(dx*cp+dy*sp,2))/thisR;
            double dRdPh = st*st*(dx*cp+dy*sp)*(dx*sp-dy*cp)/thisR;

            // Derivatives of lambdas with respect to R now.
            // Note that dlmu/dR = dlmu0/dR * ct + lmu0 * dcos(theta)/dR 
            // ---------------------------------------------------------
            double dlMu0dR, dlE0dR;
            if (isGamma) { // gamma HYP
                dlMu0dR = MuFromG(ie, thisR, 1);
                dlE0dR  = EFromG(ie, thisR, 1);
            } else { // proton HYP
                dlMu0dR = MuFromP(ie, thisR, 1); 
                dlE0dR  = EFromP(ie, thisR, 1); 
            }

            // Compute derivative of logL with respect to R.
            // Since log L = -lambda0*ct +N*log lambda0 + N*ct + cost,
            // dlogL/dR = -ct*dl0dR + N/lambda0 * dl0dR
            // --------------------------------------------
            double dlogLdR = - dlMu0dR*ct - dlE0dR*ct;
            if (lambdaMu0>0.) dlogLdR += Nmu[id]/lambdaMu0 * dlMu0dR;  
            if (lambdaE0>0.)  dlogLdR += Ne[id]/lambdaE0 * dlE0dR;  
            
            // Finally get dlogL/dx and dy from dlogLdR
            // ----------------------------------------
            dlogLdX0 += dRdX0 * dlogLdR;
            dlogLdY0 += dRdY0 * dlogLdR;

            // Also get dlogL/dtheta and dlogL/dphi
            // ------------------------------------
            dlogLdTh += (lambdaMu+lambdaE-Nmu[id]-Ne[id])*st/ct;
            if (lambdaMu0>0.) dlogLdTh += (Nmu[id]/lambdaMu0-ct)*dlMu0dR*dRdTh;
            if (lambdaE0>0.)  dlogLdTh += (Ne[id]/lambdaE0-ct)*dlE0dR*dRdTh;
            dlogLdPh += dRdPh * dlogLdR;

        } // end id loop on Nunits

        // Take a step in X0, Y0
        // ---------------------
        if (!usetrueXY ) { // otherwise we need not search for it, as the loop only is to compute logL
            currentX0 += dlogLdX0 * LearningRateX;
            currentY0 += dlogLdY0 * LearningRateY;
        }
        StepsizeX->Fill((float)istep,fabs(dlogLdX0*LearningRateX));
        StepsizeY->Fill((float)istep,fabs(dlogLdY0*LearningRateY));

        // Also take a step in theta and phi
        // ---------------------------------
        if (!usetrueAngs) {
            double dth = dlogLdTh * LearningRateTh;
            double dph = dlogLdPh * LearningRatePh;
            currentTheta += dth;
            currentPhi   += dph;
            if (currentTheta>=pi/2.) currentTheta = pi/2-epsilon; // hard reset if hitting boundary
            if (currentTheta<0.) currentTheta = 0.;               // hard reset if hitting boundary
            // For the logic of the program, phi or phi+pi do not change matters. When we introduce timing this will have to change
            currentPhi = fmod(currentPhi,pi);
            if (currentPhi<0.) currentPhi+=pi;
        }
        StepsizeT->Fill((float)istep,fabs(dlogLdTh * LearningRateTh));
        StepsizeP->Fill((float)istep,fabs(dlogLdPh * LearningRatePh));

        double RateModPl = 1.1;
        double RateModMi = 1/1.1; 
        if (dlogLdX0 * prevdlogLdX0<0.) {
            LearningRateX *= RateModMi;  
        } else {
            LearningRateX *= RateModPl; 
        }
        if (dlogLdY0 * prevdlogLdY0<0.) {
            LearningRateY *= RateModMi; 
        } else {
            LearningRateY *= RateModPl; 
        }
        if (dlogLdTh * prevdlogLdTh<0.) {
            LearningRateTh *= RateModMi; 
        } else {
            LearningRateTh *= RateModPl; 
        }
        if (dlogLdPh * prevdlogLdPh<0.) {
            LearningRatePh *= RateModMi; 
        } else {
            LearningRatePh *= RateModPl; 
        }

        prevdlogLdX0 = dlogLdX0;
        prevdlogLdY0 = dlogLdY0;
        prevdlogLdTh = dlogLdTh;
        prevdlogLdPh = dlogLdPh;
        istep++;

        DL->Fill((float)istep,logL-prevlogL);

        // if (debug) cout << "     " << isGamma << " step = " << istep << " X0=" << currentX0 << "," << currentY0 << " true= " << TrueX0[is] << " " << TrueY0[is] 
        //                 << " th,ph = " << currentTheta << " " << currentPhi << " true = " << TrueTheta[is] << " " << TruePhi[is] << " LL = " << logL << " prev = " << prevlogL << endl;
    } while (istep<Nsteps); 
 
    // if (debug) cout << "  isGamma = " << isGamma << "istep = " << istep << " Lik = " 
    //                 << logL << " x,y = " << currentX0 << "," << currentY0 << " Th = " << currentTheta << " Ph = " << currentPhi << " truexy = " 
    //                 << TrueX0[is] << "," << TrueY0[is] << " true th = " << TrueTheta[is] << " true ph = " << TruePhi[is] << endl;

    // Now we have the estimates of X0, Y0, and the logLR at max for event is 
    // ----------------------------------------------------------------------
    if (isGamma && IsGamma[ie][is]) { // gamma
        double delta = currentX0-TrueX0[is];
        if (fabs(delta)<maxdxy) {
            DXG->Fill(delta);
        } else {
            if (delta>0.) DXG->Fill(maxdxy-epsilon);
            if (delta<0.) DXG->Fill(-maxdxy+epsilon);
        }
        delta = currentY0-TrueY0[is];
        if (fabs(delta)<maxdxy) {
            DYG->Fill(delta);
        } else {
            if (delta>0.) DYG->Fill(maxdxy-epsilon);
            if (delta<0.) DYG->Fill(-maxdxy+epsilon);
        }
        DTHG->Fill(currentTheta-TrueTheta[is]);
        DTHGvsT->Fill(currentTheta-TrueTheta[is],TrueTheta[is]);
        DPHG->Fill(currentPhi-TruePhi[is]);
    } else if (!isGamma && !IsGamma[ie][is]) { // proton
        double delta = currentX0-TrueX0[is];
        if (fabs(delta)<maxdxy) {
            DXP->Fill(delta);
        } else { 
            if (delta>0.) DXP->Fill(maxdxy-epsilon);
            if (delta<0.) DXP->Fill(-maxdxy+epsilon);
        }
        delta = currentY0-TrueY0[is];
        if (fabs(delta)<maxdxy) {
            DYP->Fill(delta);
        } else {
            if (delta>0.) DYP->Fill(maxdxy-epsilon);
            if (delta<0.) DYP->Fill(-maxdxy+epsilon);
        }
        DTHP->Fill(currentTheta-TrueTheta[is]);
        DTHPvsT->Fill(currentTheta-TrueTheta[is],TrueTheta[is]);
        DPHP->Fill(currentPhi-TruePhi[is]);
    }
    if (isGamma) {
        x0meas[ie][is][0] = currentX0;
        y0meas[ie][is][0] = currentY0;
        thmeas[ie][is][0] = currentTheta;
        phmeas[ie][is][0] = currentPhi;
    } else {
        x0meas[ie][is][1] = currentX0;
        y0meas[ie][is][1] = currentY0;
        thmeas[ie][is][1] = currentTheta;
        phmeas[ie][is][1] = currentPhi;
    }
    if (logL!=logL) {
        cout << "Problems with logL, return -largenumber" << endl;
        logL = -largenumber;
        SaveLayout();
        return 0.;
    }
    return logL;
}

// Compute log likelihood ratio test statistic for one shower, by
// finding max value vs X0,Y0,theta,phi of shower for both hypotheses
// ------------------------------------------------------------------
void FindLogLR (int ie, int is) { 

    double logLG = FindShowerPos (ie,is,true);  // Find shower position by max lik of gamma hypothesis
    double refX0g = x0meas[ie][is][0];
    double refY0g = y0meas[ie][is][0];
    double refThg = thmeas[ie][is][0];
    double refPhg = phmeas[ie][is][0];
    double logLP = FindShowerPos (ie,is,false); // Find shower position by max lik of proton hypothesis
    double refX0p = x0meas[ie][is][1];
    double refY0p = y0meas[ie][is][1];
    double refThp = thmeas[ie][is][1];
    double refPhp = phmeas[ie][is][1];
    logLRT[ie][is] = logLG-logLP;
    if (debug) cout << "shower " << is << "Logs = " << logLG << " " << logLP << " ratio = " << logLRT[ie][is] << endl;

    // NB below we sample the Poissons, hence we lose the ability to check the variation of the utility for same set
    // of showers (see at the end of SGD loop), which is a useful debugging metric. So to do that,
    // we need to bypass the loop below, by putting a kludge on sigma2lrt.
    // ---------------------------------------------------------------------------------------------------
    //if (speedup) {
    //cout << logLRT[ie][is] << endl;
    //sigma2LRT[ie][is] = 40000.; ///////////////////////////////////// K L U D G E
    //return;
    //}

    // We try to compute sigma2lrt by getting as max lik estimates the estimates for true position parameters
    // ------------------------------------------------------------------------------------------------------
    for (int id=0; id<Nunits; id++) {
        double Nmuup = 0.;
        double Nmudo = 0.;
        double Neup  = 0.;
        double Nedo  = 0.;
        if (IsGamma[ie][is]) {
            if (mug[id]>0.) {
                Nmuup = myRNG->Poisson(mug[id]+sqrt(mug[id])); // Resample to get variation 
                Nmudo = myRNG->Poisson(mug[id]-sqrt(mug[id]));
            }
            if (eg[id]>0.) {
                Neup  = myRNG->Poisson(eg[id]+sqrt(eg[id]));  // Resample to get variation
                Nedo  = myRNG->Poisson(eg[id]-sqrt(eg[id]));
            }
        } else {
            if (mup[id]>0.) {
                Nmuup = myRNG->Poisson(mup[id]+sqrt(mup[id])); // Resample to get variation 
                Nmudo = myRNG->Poisson(mup[id]-sqrt(mup[id]));
            }
            if (ep[id]>0.) {
                Neup  = myRNG->Poisson(ep[id]+sqrt(ep[id]));  // Resample to get variation
                Nedo  = myRNG->Poisson(ep[id]-sqrt(ep[id]));
            }
        }
    }

    // Repeat a few times to get a hunch of sigma2
    // -------------------------------------------
    double avelogLRT  = 0.;
    double avelogLRT2 = 0.;
    for (int rep=0; rep<Nrep4sigma; rep++) {

        // We need to compute the variability of the logLR for this event, which is either a gamma or a proton.
        // Hence we rely on the expected values of fluxes depending on which is the correct hypothesis
        // ----------------------------------------------------------------------------------------------------
        for (int id=0; id<Nunits; id++) {
            Nmu[id] = 0.;
            Ne[id]  = 0.;
            if (IsGamma[ie][is]) {
                if (mug[id]>0.) Nmu[id] = myRNG->Poisson(mug[id]); // Resample to get variation 
                if (eg[id]>0.) Ne[id]   = myRNG->Poisson(eg[id]);  // Resample to get variation
            } else {
                if (mup[id]>0.) Nmu[id] = myRNG->Poisson(mup[id]); // Resample to get variation
                if (ep[id]>0.)  Ne[id]  = myRNG->Poisson(ep[id]);  // Resample to get variation
            }
        }
        logLG = FindShowerPos(ie,is,true);
        DX0g->Fill(refX0g-x0meas[ie][is][0]);
        DY0g->Fill(refY0g-y0meas[ie][is][0]);
        DThg->Fill(refThg-thmeas[ie][is][0]);
        DPhg->Fill(pi-fabs(fabs(refPhg-phmeas[ie][is][0])-pi));
        logLP = FindShowerPos(ie,is,false);
        DX0p->Fill(refX0p-x0meas[ie][is][1]);
        DY0p->Fill(refY0p-y0meas[ie][is][1]);
        DThp->Fill(refThp-thmeas[ie][is][1]);
        DPhp->Fill(pi-fabs(fabs(refPhp-phmeas[ie][is][1])-pi));
        avelogLRT  += logLG-logLP;
        avelogLRT2 += pow(logLG-logLP,2);

    }
    avelogLRT  = avelogLRT/Nrep4sigma;
    avelogLRT2 = avelogLRT2/Nrep4sigma;
    sigma2LRT[ie][is] = avelogLRT2-pow(avelogLRT,2); 
    if (sigma2LRT[ie][is]<=0.) {
        sigma2LRT[ie][is] = epsilon+fabs(avelogLRT); // Arbitrary - see if it works, for now
    } 
    SigLRT->Fill(sqrt(sigma2LRT[ie][is]));
     double DR;
    if (IsGamma[ie][is]) {
        DR = sqrt(pow(TrueX0[is]-refX0g,2)+pow(TrueY0[is]-refY0g,2));
    } else {
        DR = sqrt(pow(TrueX0[is]-refX0p,2)+pow(TrueY0[is]-refY0p,2));
    }
    SigLvsDR->Fill(sqrt(sigma2LRT[ie][is]),DR);

    return;
}

// Calculation of dlogLR over dR
// -----------------------------
double dlogLR_dR (int id, int ie, int is) {

    // Note: in this calculation one might be tempted of using true X0, Y0 positions
    // (in the logic of the Asimov approximation). However, this would abstract away
    // from the calculation of the LLR and extraction of the shower positions, which
    // is performed here with the maximum power (Neyman-Pearson's lemma) from the two
    // simple hypotheses. So we need to include that modeling part in the optimization.
    // Also note: we take x0meas, y0meas as the values that maximize the log L for the
    // considered hypothesis. The logLR we are differentiating is a ratio of profiles
    // over those values, so we need to take the corresponding R values.
    // -------------------------------------------------------------------------------
    double thisRg = EffectiveDistance(x[id],y[id],x0meas[ie][is][0],y0meas[ie][is][0],thmeas[ie][is][0],phmeas[ie][is][0],0);
    double thisRp = EffectiveDistance(x[id],y[id],x0meas[ie][is][1],y0meas[ie][is][1],thmeas[ie][is][1],phmeas[ie][is][1],0);
    if (thisRg<Rmin) thisRg = Rmin; // for consistency with rest of code
    if (thisRp<Rmin) thisRp = Rmin; // idem
    
    // Mean values
    // -----------
    double lambdaMuG, lambdaEG;
    double lambdaMuP, lambdaEP;
    lambdaMuG = MuFromG(ie, thisRg, 0)*cos(thmeas[ie][is][0]);
    lambdaEG  = EFromG(ie,  thisRg, 0)*cos(thmeas[ie][is][0]);
    lambdaMuP = MuFromP(ie, thisRp, 0)*cos(thmeas[ie][is][1]);
    lambdaEP  = EFromP(ie,  thisRp, 0)*cos(thmeas[ie][is][1]);
    
    // Derivatives with respect to R now
    // ---------------------------------
    double dlMuGdR, dlEGdR;
    double dlMuPdR, dlEPdR;
    dlMuGdR = MuFromG(ie, thisRg, 1)*cos(thmeas[ie][is][0]);
    dlEGdR  = EFromG(ie,  thisRg, 1)*cos(thmeas[ie][is][0]);
    dlMuPdR = MuFromP(ie, thisRp, 1)*cos(thmeas[ie][is][1]);
    dlEPdR  = EFromP(ie,  thisRp, 1)*cos(thmeas[ie][is][1]);

    // The calculation goes as follows, for an event k and detector unit i:
    // logLG = {-lambda_mug_i - lambda_eg_i + N_mug_i*log(lambda_mug_i) +
    //         N_eg_i*log(lambda_eg_i) -log(N_mu_i!) - log(N_e_i!) }
    // from which we get:
    // dlogLG/dR = { -dlambda_mug_i/dR_i - dlambda_eg_i/dR_i +d/dR_i(N_mug_i*log(lambda_mug_i)) +
    //             d/dR_i(N_eg_i*log(lambda_eg_i)) + d/dR_i(-log(N_mu_i!)) + d_dR_i(-log(N_mu_i!)) }
    // dlogLP/dR = { -dlambda_mup_i/dR_i - dlambda_ep_i/dR_i +d/dR_i(N_mup_i*log(lambda_mup_i)) +
    //             d/dR_i(N_ep_i*log(lambda_ep_i)) + d/dR_i(-log(N_mup_i!)) + d_dR_i(-log(N_mup_i!)) }
    // -----------------------------------------------------------------------------------------------
    // Now in Asimov approx, lambda is equal to N, and derivatives also are equal. 
    // Apart from the factorial terms we then get:
    // dlogLR/dR = { log(lambda_mug_i)*dlambda_mug_i/dR_i + log(lambda_eg_i)*dlambda_eg_i/dR_i +
    //               -log(lambda_mup_i)*dlambda_mup_i/dR_i - log(lambda_ep_i)*dlambda_ep_i/dR_i }
    // As for the factorials: 
    // d/dx(log(N(x)!)) = 1/N(x)! * d(N(x)!)/dx = -gammaEM + sum_k^N(x) (1/k) 
    // --> d/dR_i(-log(N_mu_i!)) = dN_mu_i/dR_i * (-gammaEM + sum_k^N_mu_i (1/k))
    // Note though, that they only play in if we take the Asimov approximation, as in that case we
    // have the equality lambda = N and lambda changes at numerator (gammas) and denominator (protons)
    // in the logLR; in the case of no approximation, the numbers are the same at num and den, and cancel!
    // ---------------------------------------------------------------------------------------------------
    // double sumkmug = -gamma_EM;
    // double sumkeg  = -gamma_EM;
    // double sumkmup = -gamma_EM;
    // double sumkep  = -gamma_EM;
    //  if (Asimov) {
        /*
        for (int k=1; k<lambdaMuG; k++) { // We sum to lambdaMuG === N_mug_i in Asimov approx
            sumkmug += 1./k;
        }
        for (int k=1; k<lambdaEG; k++) {
            sumkeg += 1./k;
        }
        for (int k=1; k<lambdaMuP; k++) {
            sumkmup += 1./k;
        }
        for (int k=1; k<lambdaEP; k++) {
            sumkep += 1./k;
        }
        */
        // A good approx to these sums is log(k) + 1/2k + gamma_EM
        // -------------------------------------------------------
        // if (lambdaMuG>1.) sumkmug += log((double)((int)lambdaMuG)) + 1./(2.*(int)lambdaMuG) + gamma_EM;
        // if (lambdaEG>1.)  sumkeg  += log((double)((int)lambdaEG))  + 1./(2.*(int)lambdaEG)  + gamma_EM;
        // if (lambdaMuP>1.) sumkmup += log((double)((int)lambdaMuP)) + 1./(2.*(int)lambdaMuP) + gamma_EM;
        // if (lambdaEP>1.)  sumkep  += log((double)((int)lambdaEP))  + 1./(2.*(int)lambdaEP)  + gamma_EM;
    //  } 
    double dlogLGdR = -dlMuGdR-dlEGdR;
    double dlogLPdR = -dlMuPdR-dlEPdR;

    // Always take the Asimov approximation here - for now -, as it is memory-expensive to use 
    // Nmu[id], Ne[id] in this calculation (requires to keep as static a [i][j][k] matrix of dllR_dR values)
    // -----------------------------------------------------------------------------------------------------
    // if (Asimov) {
        /*
        if (lambdaMuG>0. && lambdaEG>0.) {
            dlogLGdR = log(lambdaMuG)*dlMuGdR + log(lambdaEG)*dlEGdR - dlMuGdR*sumkmug - dlEGdR*sumkeg;
        } else {
            dlogLGdR = -largenumber;
        }
        if (lambdaMuP>0. && lambdaEP>0.) {
            dlogLPdR = log(lambdaMuP)*dlMuPdR + log(lambdaEP)*dlEPdR - dlMuPdR*sumkmup - dlEPdR*sumkep;
        } else {
            dlogLPdR = -largenumber;
        }
        */
    // } else {

    if (lambdaMuG>0.) {
        dlogLGdR += (Nmu[id]/lambdaMuG + log(lambdaMuG))*dlMuGdR;
    }
    if (lambdaEG>0.) {
        dlogLGdR += (Ne[id]/lambdaEG+log(lambdaEG))*dlEGdR;
    }
    if (lambdaMuP>0.) {
        dlogLPdR += (Nmu[id]/lambdaMuP + log(lambdaMuP))*dlMuPdR;
    }
    if (lambdaEP>0.) {
        dlogLPdR += (Ne[id]/lambdaEP+log(lambdaEP))*dlEPdR;
    }
    //}

    if (dlogLGdR!=dlogLGdR || dlogLPdR!=dlogLPdR) {
        cout << "Trouble in dlogLR_dR" << endl;
        SaveLayout();
        return 0.;
    }
    return dlogLGdR-dlogLPdR;
}

// This routine finds the RCF bound on the variance of the signal fraction that can be 
// extracted from a 2-component fit to the LLR distributions
// -----------------------------------------------------------------------------------
double VarianceGammaFraction (int ie) {

    double Fg = TrueGammaFraction[ie]; // we Asimovize it 
    double inv_variance = 0.;
    // To compute the variance of a signal fraction extracted from the fit to the LLR
    // distribution of a batch of data, we need the pdf of gamma and proton LLR values.
    // We get those from the first Nevents, by adding up the contribution of each
    // event to the value we need (x, see below) assumed to be sampled from a Gaussian
    // resolution spread.
    // --------------------------------------------------------------------------------
    JS = 0.;
    for (int is=Nevents; is<Nevents+Nbatch; is++) {
        double x = logLRT[ie][is];
        double g_x = 0.;
        double p_x = 0.;
        for (int it=0; it<Nevents; it++) {
            double t = logLRT[ie][it];
            double sigma_t = sqrt(sigma2LRT[ie][it]); 
            double y = exp(-pow((x-t)/(2.*sigma_t),2))/(sqrt2pi*sigma_t);
            if (IsGamma[ie][it]) {
                g_x += y;
            } else {
                p_x += y;
            }
        }

        // Get normalized densities
        // ------------------------
        g_x = g_x/(Fg*Nevents);
        p_x = p_x/((1.-Fg)*Nevents);

        // Also compute JS divergence for a check
        // --------------------------------------
        double m_x = 0.5*(g_x+p_x);
        if (g_x>0. && p_x>0.) JS += 0.5 * (g_x*log(g_x/m_x)+p_x*log(p_x/m_x));

        // Use RCF to get variance
        // -----------------------
        if (g_x==0. && p_x==0.) continue;
        inv_variance += pow((g_x-p_x)/(Fg*g_x+(1.-Fg)*p_x),2);
        if (debug) cout << "is = " << is << " inv_variance = " << inv_variance << " g_x, p_x = " << g_x << " " << p_x << endl;
    }

    if (inv_variance==0.) return largenumber;
    double vgf = 1./inv_variance;
    if (vgf==0. || vgf!=vgf) vgf = epsilon;
    return vgf;
}

// Compute utility function and gradient
// -------------------------------------
void ComputeUtility (int NEvalues) {
    Utility = 0.;
    UtilityErr = 0.;
    double Utilerr2 = 0.;
    for (int ie=0; ie<NEvalues; ie++) {
        Utility += TrueGammaFraction[ie]/GammaFracErr[ie] * weight[ie];
        Utilerr2 += pow(weight[ie]/GammaFracErr[ie],2);
    }
    UtilityErr = sqrt(Utilerr2);
    return;
}

// Sort LLRT values
// ----------------
void SortLLRT (int N) {
    for (int j=0; j<NEvalues; j++) {
        for (int times=0; times<N; times++) {
            for (int i=N-1; i>0; i--) {
                if (logLRT[j][i]<logLRT[j][i-1]) {
                    double tmp = logLRT[j][i-1];
                    bool kind  = IsGamma[j][i-1];
                    logLRT[j][i-1] = logLRT[j][i];
                    IsGamma[j][i-1] = IsGamma[j][i];
                    logLRT[j][i] = tmp;
                    IsGamma[j][i] = kind;
                }
            }
        }
    }
    return;
}


// Function that fills parameters of showers
// -----------------------------------------
void ReadShowers () {
	double Qparam_mufromp[5][3] = {{1000, 2.6, 500},{200,  2.5, 500},{50,   2.1, 500},{5.0, 2.1, 500},{0.5,  2.0, 500}};
        double Qparam_efromp[5][3] = {{ 1000000, 2.85, 2000,},{ 200000,  2.8,  2000},{30000,   2.75, 2000},{1500,    2.7,  2000},{ 20,  2.4,  2000}};
        double Qparam_mufromg[5][3] = {{ 80,    2.7,  200},{ 7,     2.3,  200},{0.4,   1.75, 200},{0.04,  1.6,  200},{0.002, 1.4,  200,}};
        double Qparam_efromg[5][3] = {{3000000, 3.2, 100},{300000,  3,   100},{40000,   2.9, 100},{ 500,     2.5, 400},{ 20,      2.5, 400}};
    for(int i = 0; i <5 ; i++)
    {
        // inner for loop to traverse column
        for(int j = 0; j < 3; j++)
        {
            // insert arr[row][col] to transpose[col][row]
            param_mufromp[j][i] = Qparam_mufromp[i][j];
            param_efromp[j][i] = Qparam_efromp[i][j];
            param_mufromg[j][i] = Qparam_mufromg[i][j];
            param_efromg[j][i] = Qparam_efromp[i][j];
        }
    }
    /*string trainPath  = "./MODE/";
    ifstream asciifile;
    std::stringstream sstr2;
    sstr2 << "swgoshowers";
    string traininglist = trainPath  + sstr2.str() + ".txt";
    asciifile.open(traininglist);

    if (debug) cout << "Energy:" << endl;
    for (int ie=0; ie<NEvalues; ie++) {
        if (debug) cout << "  ie = " << ie << " E = "; 
        double e;
        asciifile >> e;
        if (debug) cout << e << " Pars MuP = ";
        for (int ip=0; ip<3; ip++) {
            asciifile >> e;
            param_mufromp[ip][ie] = e;
            if (debug) cout << e << " ";
        }
        if (debug) cout << " Pars EP = ";
        for (int ip=0; ip<3; ip++) {
            asciifile >> e;
            param_efromp[ip][ie] = e;
            if (debug) cout << e << " ";
        }
        if (debug) cout << " Pars MuG = ";
        for (int ip=0; ip<3; ip++) {
            asciifile >> e;
            param_mufromg[ip][ie] = e;
            if (debug) cout << e << " ";
        }
        if (debug) cout << " Pars EG = ";
        for (int ip=0; ip<3; ip++) {
            asciifile >> e;
            param_efromg[ip][ie] = e;
            if (debug) cout << e << " ";
        }
        if (debug) cout << endl;
    }
    asciifile.close();*/
    return;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Main routine
//
// ----------------------------------------------------------------------------------------------------------------------------------------------

void swgolo (int Nev=1000, int Nba=1000, int Nu=1000, int NE=5, int Nep=100, 
             double DetectorSpacing=10., double SpacingStep = 1., double Rsl=100., int sh=2, 
             int cm=0, int Nst=30, double pr=1., double pra=1.) {


    // Pass parameters:
    // ----------------
    // Nevents   = number of generated shower for templates
    // Nbatch    = number of showers per batch
    // Nunits    = number of detector elements
    // NEvalues  = number of energy values of generated showers
    // Nepochs   = number of SGD loops
    // DetectorSpacing  = initial spacing of tanks
    // SpacingStep = increase in spacing 
    // Rslack      = space of showers away from detector units
    // shape       = geometry of the initial layout (0=hexagonal, 1=taxi, 2=spiral)
    // commonMode  = whether xy of units is varied independently (0), or radius (1) or offset (2)
    // Nsteps      = number of steps of position likelihood finding
    // posrate     = multiplier of learning rate in position for shower reco likelihood
    // posrateang  = multiplier of learning rate in angle for shower reco likelihood

    // UNITS
    // -----
    // position: meters
    // angle:    radians
    // time:     seconds
    // energy:   PeV

    // Get static values from pass parameters
    // --------------------------------------
    Nevents    = Nev;
    Nepochs    = Nep;
    Nbatch     = Nba;
    Nunits     = Nu;
    NEvalues   = NE;
    Rslack     = Rsl;
    shape      = sh;
    commonMode = cm;
    Nsteps     = Nst;
    posrate    = pr;
    posrateang = pra;

    // Safety checks
    // -------------
    if (Nunits>maxUnits) {
        cout << "Too many units. Stopping." << endl;
        return;
    }
    if (Nevents+Nbatch>maxEvents) {
        cout << "Too many events. Stopping." << endl;
        return;
    }
    if (NEvalues>maxEbins) {
        cout << "Too many E bins. Stopping." << endl;
        return; 
    }
    if (Nepochs>maxEpochs) {
        cout << "Too many epochs. Stopping." << endl;
        return;
    }
    if (DetectorSpacing<=0.) {
        cout << "DetectorSpacing must be >0. Stopping." << endl;
        return;
    }

    // Other checks
    // ------------
    if (shape==3 && SpacingStep==0.) {
        cout << "Sorry, for circular shapes you need to define a radius increment larger than zero!" << endl;
        return;
    }
    cout << endl;
    cout << "     ****************************************************************" << endl;
    cout << endl;
    cout << "                          S   W   G   O   L   O                      " << endl;
    cout << endl; 
    cout << "         Southern Wide-field Gamma Observatory Layout Optimization   " << endl;
    cout << endl;
    cout << "         Proof-of-principle study                                    " << endl;
    cout << "         of SWGO detector layout optimization                        " << endl;
    cout << endl;
    cout << "                                            T. Dorigo, Oct 22 2022   " << endl;
    cout << endl;
    cout << "     ****************************************************************" << endl;
    cout << endl;

    gStyle->SetOptStat(0);
    gStyle->SetOptFit(0);

    // Get a sound RN generator
    // ------------------------
    delete gRandom;
    static TRandom3 * myRNG = new TRandom3();
 
    // Define the current geometry 
    // ---------------------------
    if (readGeom) {
        ReadLayout ();
    } else {
        DefineLayout(DetectorSpacing,SpacingStep);
    }

    // Define number of R bins depending on spanR (which is now defined) and Rslack
    // ----------------------------------------------------------------------------
    NRbins = 100*(spanR+2*Rslack)/spanR;
    if (NRbins>maxRbins) NRbins=maxRbins;

    // Read in parametrizations of particle fluxes
    // -------------------------------------------
    ReadShowers ();

    // Define weights for Utility function
    // -----------------------------------
    for (int ie=0; ie<NEvalues; ie++) {
        weight[ie] = 1./NEvalues;
    }

    // Big optimization loop, modifying detector layout
    // ------------------------------------------------
    cout << "     Starting gradient descent loop " << endl;
    double maxUtility = 0.;
    int imax = 0;

    // Histogram definition
    // --------------------
    TH1D * U        = new TH1D     ("U",      "", Nepochs, 0.5, (double)Nepochs+0.5);  
    TH1D * JSSum    = new TH1D     ("JSSum",  "", Nepochs, 0.5, (double)Nepochs+0.5);  
    TProfile * Uave = new TProfile ("Uave",   "", Nepochs/10, 0.5, (double)Nepochs+0.5,0.,100.);  
    TH1D * GFmeas   = new TH1D     ("GFmeas", "", 5, -3.5, 1.5);
    TH1D * GFtrue   = new TH1D     ("GFtrue", "", 5, -3.5, 1.5);

    // After calling DefineLayout we can define the layout plots
    // ---------------------------------------------------------
    TH2D * Layout   = new TH2D ("Layout",   "", 500, -spanR-2*Rslack, spanR+2*Rslack, 500, -spanR-2*Rslack, spanR+2*Rslack);
    TH2D * Layout2  = new TH2D ("Layout2",  "", 500, -0.5*(spanR+2*Rslack), 0.5*(spanR+2*Rslack), 500, -0.5*(spanR+2*Rslack), 0.5*(spanR+2*Rslack));
    TH2D * Showers  = new TH2D ("Showers",  "", 500, -spanR-2*Rslack, spanR+2*Rslack, 500, -spanR-2*Rslack, spanR+2*Rslack);
    TH2D * Showers2 = new TH2D ("Showers2", "", 500, -0.5*(spanR+2*Rslack), 0.5*(spanR+2*Rslack), 500, -0.5*(spanR+2*Rslack), 0.5*(spanR+2*Rslack));
    TH2D * Showers3 = new TH2D ("Showers3", "", 200, -spanR-2*Rslack, spanR+2*Rslack, 200, -spanR-2*Rslack, spanR+2*Rslack);
    int NbinsRdistr = Nunits/5;
    if (NbinsRdistr<10) NbinsRdistr = 20;
    TH1D * Rdistr0      = new TH1D     ("Rdistr0",  "", NbinsRdistr, 0., spanR+2*Rslack);
    TH1D * Rdistr       = new TH1D     ("Rdistr",   "", NbinsRdistr, 0., spanR+2*Rslack);
    TH1D * MutualD      = new TH1D     ("MutualD",  "", 400, 0., 2*(spanR+2*Rslack));
    TH2D * LLRvsD       = new TH2D     ("LLRvsD",   "", 100, -100000,100000, 100, 0., 2*(spanR+2*Rslack));
    TH1D * LLRP         = new TH1D     ("LLRP",     "", 1000,-20000000.,20000000.); 
    TH1D * LLRG         = new TH1D     ("LLRG",     "", 1000,-20000000.,20000000.);
    TH1D * LLRPb        = new TH1D     ("LLRPb",    "", 1000,-20000000.,20000000.);
    TH1D * LLRGb        = new TH1D     ("LLRGb",    "", 1000,-20000000.,20000000.);
    TProfile * dUdx     = new TProfile ("dUdx",     "", Nepochs, -0.5, Nepochs-0.5, 0., 100.);
    TH1D * PosQ         = new TH1D     ("PosQ",     "", Nepochs, -0.5, Nepochs-0.5);
    TH1D * AngQ         = new TH1D     ("AngQ",     "", Nepochs, -0.5, Nepochs-0.5);
    // TH1D * CosDir       = new TH1D     ("CosDir",   "", 100, -1., 1.);
    // TProfile * CosvsEp  = new TProfile ("CosvsEp",  "", Nepochs, -0.5, Nepochs-0.5, -1., 1.);
    TH2D * LR           = new TH2D     ("LR",       "", Nepochs, -0.5, Nepochs-0.5, 100, -10., 15.);
    float size = 0.5;
    Layout->SetMarkerStyle(20);
    Layout->SetMarkerColor(kBlack);
    Layout->SetMarkerSize(size);
    Layout2->SetMarkerStyle(20);
    Layout2->SetMarkerColor(kBlack);
    Layout2->SetMarkerSize(size);
    Showers->SetMarkerStyle(24);
    Showers2->SetMarkerStyle(24);
    Showers->SetMarkerSize(size);
    Showers->SetMarkerColor(kRed);
    Showers2->SetMarkerSize(size);
    Showers2->SetMarkerColor(kRed);
    Rdistr0->SetLineColor(kRed);
    Rdistr0->SetLineWidth(3);

    for (int id=0; id<Nunits; id++) {
        Layout->Fill(x[id],y[id]);
        Layout2->Fill(x[id],y[id]);
        Rdistr0->Fill(sqrt(x[id]*x[id]+y[id]*y[id]));
    }

    // If shower positions are fixed, we get that done as early as now
    // Note that since IsGamma[] indicates a photon for even is, and
    // a proton for odd is (see below when GenerateShower is called),
    // we are alternating photons and protons on the same radii. This
    // must be changed if other geometries are concerned, in case it
    // may interfere with correct placement of detector units.
    // ---------------------------------------------------------------
    if (fixShowerPos) {
        if (hexaShowers) {
            double r0 = sqrt(pow(spanR+Rslack,2)*pi/Nevents);
            TrueX0[0] = Xoffset;
            TrueY0[0] = Yoffset;
            int is = 1;
            int n = 6;
            double r = r0;
            do {
                for (int ith=0; ith<n && is<Nevents; ith++) {
                    double theta = ith*2.*pi/n;
                    TrueX0[is] = Xoffset + r*cos(theta);
                    TrueY0[is] = Yoffset + r*sin(theta);
                    if (debug) cout << is << " ith=" << ith << " r=" << r << " x,y= " << TrueX0[is] << " " << TrueY0[is] << endl;
                    is++;
                }
                n += 6;
                r += r0;
            } while (is<Nevents);

            // And now do the same for the Nbatch events
            // -----------------------------------------
            TrueX0[Nevents] = Xoffset;
            TrueY0[Nevents] = Yoffset;
            is = Nevents;
            n = 6;
            r = r0;
            do {
                for (int ith=0; ith<n && is<Nevents+Nbatch; ith++) {
                    double theta = ith*2.*pi/n;
                    TrueX0[is] = Xoffset + r*cos(theta);
                    TrueY0[is] = Yoffset + r*sin(theta);
                    if (debug) cout << is << " ith=" << ith << " r=" << r << " x,y= " << TrueX0[is] << " " << TrueY0[is] << endl;
                    is++;
                }
                n += 6;
                r += r0;
            } while (is<Nevents+Nbatch);
 
        } else {

            // Below for a square grid of showers
            // ----------------------------------
            int side = sqrt(Nevents);
            for (int is=0; is<Nevents; is++) {
                TrueX0[is] = Xoffset -spanR -Rslack + 2.*(spanR+Rslack)*(is%side+0.5)/side;
                TrueY0[is] = Yoffset -spanR -Rslack + 2.*(spanR+Rslack)*(is/side+0.5)/side;
                if (debug) cout << is << " x,y= " << TrueX0[is] << " " << TrueY0[is] << endl;
            }

            // Same, for Nbatch events 
            // -----------------------
            side = sqrt(Nbatch);
            for (int is=Nevents; is<Nevents+Nbatch; is++) {
                TrueX0[is] = Xoffset -spanR -Rslack + 2.*(spanR+Rslack)*((is-Nevents)%side+0.5)/side;
                TrueY0[is] = Yoffset -spanR -Rslack + 2.*(spanR+Rslack)*((is-Nevents)/side+0.5)/side;
                if (debug) cout << is << " x,y= " << TrueX0[is] << " " << TrueY0[is] << endl;
            }
        }

        for (int is=0; is<Nevents; is++) {
            Showers->Fill(TrueX0[is],TrueY0[is]);
            Showers2->Fill(TrueX0[is],TrueY0[is]);
        }
    }

    // Define number of iterations to measure sigmaLR 
    // ----------------------------------------------
    if (speedup) {
        Nrep4sigma = 3;
        Nsteps = 10.; // coarser likelihood for shower position
    }

    // SGD stuff
    // ---------
    int epoch = 0;
    double StartLearningRate = 100.; // 0.1*DetectorSpacing; 
    double LearningRate[maxUnits];
    double LearningRateR[maxRbins];
    double LearningRateC = StartLearningRate; 
    for (int i=0; i<Nunits; i++) {
        LearningRate[i] = StartLearningRate;
    }
    for (int ir=0; ir<NRbins; ir++) {
        LearningRateR[ir] = StartLearningRate;
    }
    double maxDispl = 5.*DetectorSpacing;
    if (shape<3) {
        maxDispl = 5.*DetectorSpacing;
    } else if (shape==3) {
        maxDispl = 2.5*SpacingStep;  // max step in R during SGD
    }

    // Print canvas for temporary plots for the first time here
    // --------------------------------------------------------
    TCanvas * CT = new TCanvas ("CT","",1300,600);
    CT->Divide(4,2);
    CT->cd(1);
    U->SetMarkerStyle(20);
    U->SetLineWidth(3);
    U->SetMaximum(1.1*maxUtility);
    U->SetMinimum(0.);
    U->Draw("P");
    Uave->SetLineColor(kRed);
    Uave->Draw("SAME");
    CT->cd(2);
    Showers3->SetMarkerStyle(24);
    Showers3->SetMarkerSize(size);
    Showers3->Draw("COL4");
    Layout->Draw("SAME");
    CT->cd(3);
    Showers2->Draw();
    Layout2->Draw("SAME");
    CT->cd(4);
    Rdistr0->Draw();
    Rdistr->SetLineWidth(3);
    Rdistr->SetMinimum(0);
    Rdistr->Draw("SAME");
    CT->cd(5);
    PosQ->SetLineWidth(3);
    PosQ->Draw();
    CT->cd(6);
    AngQ->SetLineWidth(3);
    AngQ->Draw();
    //  CT->cd(7);
    //  CosDir->Draw();
    //  CT->cd(8);
    //  CosvsEp->SetLineColor(kRed);
    //  CosvsEp->SetLineWidth(3);
    //  CosvsEp->Draw();
    CT->cd(7);
    LR->Draw();
    CT->cd(8);
    JSSum->Draw();
    CT->Update();
    char namepng[40];
    sprintf (namepng,"./MODE/Layout%d.png",epoch);
    CT->Print(namepng);

    // Beginning of big optimization loop
    // ----------------------------------
    do { // SGD

        // Adjust Learning Rate with scheduling
        // ------------------------------------
        if (commonMode==2) {
            LearningRateC = LR_Scheduler(StartLearningRate,epoch);
            cout << "     New cycle: Learning rate is now " << LearningRateC << endl;
        }

        if (!fixShowerPos) {
            Showers->Reset();
            Showers2->Reset();
        }
        Showers3->Reset();
        LLRvsD->Reset();
        // CosDir->Reset();

        // Reset histograms tracking goodness of position fits
        // ---------------------------------------------------
        DXG->Reset();
        DYG->Reset();
        DXP->Reset();
        DYP->Reset();

        // Loop on energy points
        // ---------------------
        double JSsum = 0.; // sum of JS values for all energy points
        for (int ie=0; ie<NEvalues; ie++) {

            // We compute the templates of test statistic 
            // (log-likelihood ratio) for gammas and protons, and while
            // we are at it, we also generate a batch of extra Nbatch events for SGD
            // ---------------------------------------------------------------------
            cout << "     ie << " << ie << " is = ";
            for (int is=0; is<Nevents+Nbatch; is++) {
                if (is*0.01==is/100) cout << is << " ";

                // Position of center of shower
                // Please remember that it must cover instrumented area
                // for all scanned configurations!
                // We give some slack to the generated showers, such
                // that the system cannot discover that the illuminated
                // area has a step function
                // ----------------------------------------------------
                if (!fixShowerPos) {
                    double DfromCenter = sqrt(myRNG->Uniform(0., pow(spanR+Rslack,2)));
                    double theta = myRNG->Uniform(0.,2.*pi);
                    TrueX0[is] = Xoffset + DfromCenter*cos(theta);  
                    TrueY0[is] = Yoffset + DfromCenter*sin(theta);
                    Showers->Fill(TrueX0[is],TrueY0[is]);
                    Showers2->Fill(TrueX0[is],TrueY0[is]);
                }    

                // Find Nmu[], Ne[] for this event
                // -------------------------------
                if (is%2==0) { // Generate a gamma (and fill all mug, eg, mup, ep anyway)
                    GenerateShower(ie,is,true);
                    IsGamma[ie][is] = true;
                } else { // Generate a proton (and fill all mug, eg, mup, ep anyway)
                    GenerateShower(ie,is,false);
                    IsGamma[ie][is] = false;
                }
                if (debug) {
                    for (int id=0; id<Nunits; id++) { 
                        cout << "x,y = " << x[id] << " " << y[id] << " counts = " << Nmu[id] << " " << Ne[id] << endl;
                    }
                }

                // For this shower, find value of test statistic
                // NB this requires that we have x0true, y0true,
                // and the mug, eg, mup, ep values filled for this
                // event. GenerateShower takes care of the latter
                // -----------------------------------------------
                FindLogLR(ie,is); // Fills logLRT[] array and fills sigma2LRT
                
                if (epoch==Nepochs-1 && ie==0 && is<Nevents) {
                    if (is%2==0) { // gamma event
                        LLRG->Fill(logLRT[ie][is]);
                    } else { // proton event
                        LLRP->Fill(logLRT[ie][is]);
                    }
                }
                

                // Fill LLRvsD histo for template events
                // -------------------------------------
                /*
                if (is<Nevents) {
                    double minr = largenumber;
                    for (int i=0; i<Nunits; i++) {
                        double r = sqrt(pow(x[i]-TrueX0[is],2)+pow(y[i]-TrueY0[is],2));
                        if (r<minr) minr = r;
                    }
                    if (minr<Rmin) minr = Rmin;
                    LLRvsD->Fill(logLRT[ie][is],minr);
                }
                */

                // Also fill shower 2d distribution with error on position
                // -------------------------------------------------------
                double Derror;
                if (IsGamma[ie][is]) {
                    Derror = sqrt(pow(TrueX0[is]-x0meas[ie][is][0],2)+pow(TrueY0[is]-y0meas[ie][is][0],2));
                } else {
                    Derror = sqrt(pow(TrueX0[is]-x0meas[ie][is][1],2)+pow(TrueY0[is]-y0meas[ie][is][1],2));
                }
                Showers3->Fill(TrueX0[is],TrueY0[is],Derror);

            } // end is loop
            cout << endl;

            // Use true gamma fraction and determine gradient from RCF bound equation of variance of Fs
            // ----------------------------------------------------------------------------------------
            GammaFraction[ie] = TrueGammaFraction[ie];
            GammaFracErr[ie]  = sqrt(VarianceGammaFraction(ie));      
            JSsum += JS;

        } // end ie loop

        // Below we compute utility function and its gradient with respect to parameters undergoing optimization
        // -----------------------------------------------------------------------------------------------------
        ComputeUtility(NEvalues); 
        if (Utility<0.) Utility = 0.;
        if (Utility>MaxUtility) Utility = U->GetBinContent(epoch); // use previous value to avoid messing up the U graph
        U->SetBinContent(epoch+1,Utility);
        JSSum->SetBinContent(epoch+1,JSsum);
        Uave->Fill(epoch+1,Utility);
        // U->SetBinError(epoch+1,UtilityErr);
        if (Utility>maxUtility) {
            maxUtility = Utility;
            imax = epoch;
        }

        // Zero a few arrays
        // -----------------
        double aveDR = 0.; // Keep track of average displacement at each epoch
        double displ[maxRbins];
        double prev_displ[maxRbins];
        int Ndispl[maxRbins];
        for (int ir=0; ir<NRbins; ir++) {
            displ[ir]      = 0.;
            prev_displ[ir] = 0.;
            Ndispl[ir]     = 0;
        }
        double commondx = 0;
        double commondy = 0;
        double dpg_dRik[maxEvents];
        double dpp_dRik[maxEvents];
        double pg[maxEvents];
        double pp[maxEvents];
        double Momentum_coeff = 0.02; // to be optimized
        double CosThetaEff; // effective angle for successive displacements, used to update learning rate

        // Loop on detector units, to update detector positions following gradient of utility
        // ----------------------------------------------------------------------------------
        for (int i=0; i<Nunits; i++) { // Nunits; i++) {

            //if (myRNG->Uniform()>10./Nunits) continue; // only update a few 

            double dU_dxi = 0.;
            double dU_dyi = 0.;
            double xi = x[i]; // Fastens calcs
            double yi = y[i]; 

            // Loop on energy points
            // ---------------------
            for (int j=0; j<NEvalues; j++) {

                // The weights below are used to avoid doing a double cycle on "template-making" Nevents showers
                // and "batch" Nbatch showers. We use the same events (in the logic of the saturated model) for
                // obtaining the U variations with detector positions, by tracking the effect on U of variations
                // induced by each of the factors. So, e.g., the Pp(llr) distribution and the Pg(llr) distribution
                // vary if we move detectors around, and their variation only depends on the events that were used
                // to construct those PDFs, so we account only for the 0.5*Nevents gamma events in deriving dPg,
                // and similarly the other half (those from proton showers) for the dPp.
                // -----------------------------------------------------------------------------------------------
                double Weight_gamma  = TrueGammaFraction[j]/0.5;
                double Weight_proton = (1.-TrueGammaFraction[j])/0.5;
                double fg  = GammaFraction[j];
                double fge = GammaFracErr[j];
                bool nextj = false;
                if (fge==0.) {
                    cout << "Warning, fge=0" << endl;
                    fge = 1.;
                    nextj = true;
                }
                if (nextj) continue;

                // First fill the array dldrm[][] once and for all
                // -----------------------------------------------
                for (int m=0; m<Nevents; m++) {
                    dldrm[i][m] = dlogLR_dR(i,j,m);
                }

                // Now compute dpg_drj, dpp_drj for all batch events, by looping on pdfs
                // ---------------------------------------------------------------------
                double Fg = TrueGammaFraction[j];
                double inv_sigmafs = 0.;
                for (int k=Nevents; k<Nevents+Nbatch; k++) {
                    dpg_dRik[k] = 0.;
                    dpp_dRik[k] = 0.;
                    pg[k] = 0.;
                    pp[k] = 0.;
                    double dldrk  = dlogLR_dR(i,j,k);
                    // We need the coordinates of the point of closest approach of the shower axis to the detector,
                    // not the distance on the detector plane. Hence these require some calculation:
                    // --------------------------------------------------------------------------------------------
                    double x0_k;
                    double y0_k;
                    double th_k;
                    double ph_k;
                    if (IsGamma[j][k]) {
                        x0_k = x0meas[j][k][0];
                        y0_k = y0meas[j][k][0];
                        th_k = thmeas[j][k][0];
                        ph_k = phmeas[j][k][0];
                    } else {
                        x0_k = x0meas[j][k][1];
                        y0_k = y0meas[j][k][1];
                        th_k = thmeas[j][k][1];
                        ph_k = phmeas[j][k][1];
                    }
                    double Rik = EffectiveDistance(xi,yi,x0_k,y0_k,th_k,ph_k,0);
                    if (Rik<Rmin) Rik = Rmin;

                    // We also need the point on the shower axis from where x[],y[] is closest
                    // -----------------------------------------------------------------------
                    double zcl_k = (cos(ph_k)*(xi-x0_k)+sin(ph_k)*(yi-y0_k))*(cos(th_k)/(1.+pow(cos(th_k),2)));
                    double xcl_k = x0_k+zcl_k*cos(th_k)*cos(ph_k);
                    double ycl_k = y0_k+zcl_k*cos(th_k)*sin(ph_k);

                    // We have to derive G by logLRT[j][m] multiplied by dlogLRT[j][m]/dRik and 
                    // add the derivative of G by logLRT[j][k] multiplied by dlogLRT[j][k]/dRik
                    // Now, dlogLRT[j][m]/dRik is the cosine of the angle between Rim and Rik, the
                    // vectors connecting detector i to the axis of showers m and k; dlogLRT[j][k]/dRik is computed
                    // in the routine dlogLR_dR above.
                    // --------------------------------------------------------------------------------------------
                    for (int m=0; m<Nevents; m++) {
                        double sigma = sqrt(sigma2LRT[j][m]);
                        double dsigma_drik = 0.;
                        double Gden = sqrt2pi*sigma;
                        double G = exp(-pow((logLRT[j][m]-logLRT[j][k])/sigma,2)/2.) / Gden;
                        if (IsGamma[j][m]) {
                            double x0_m = x0meas[j][m][0];
                            double y0_m = y0meas[j][m][0];
                            double th_m = thmeas[j][m][0];
                            double ph_m = phmeas[j][m][0];
                            pg[k] += G;
                            double Rim = EffectiveDistance(xi,yi,x0_m,y0_m,th_m,ph_m,0);
                            if (Rim<Rmin) Rim = Rmin;
                            // To compute the cosine we need the point of closest distance to x[],y[] on the shower axis,
                            // for both k and m. This point is computed by writing the squared distance
                            // d^2 = z^2 + (xd-x0-z*costheta*cosphi)^2 + (yd-y0-z*costheta*sinphi)^2 
                            // and deriving wrt z, then equating to zero. This yields
                            // zcl = (xd-x0)cosphi + (yd-y0)sinphi]*costheta/(1+costheta^2)
                            // Then xcl and ycl can be obtained by substitution.
                            // ------------------------------------------------------------------------------------------
                            double zcl_m = (cos(ph_m)*(xi-x0_m)+sin(ph_m)*(yi-y0_m))*(cos(th_m)/(1.+pow(cos(th_m),2)));
                            double xcl_m = x0_m+zcl_m*cos(th_m)*cos(ph_m);
                            double ycl_m = y0_m+zcl_m*cos(th_m)*sin(ph_m);
                            double costh = ((xi-xcl_m)*(xi-xcl_k) + (yi-ycl_m)*(yi-ycl_k) + zcl_k*zcl_m)/ (Rik*Rim);
                            dpg_dRik[k] += G * ((logLRT[j][m]-logLRT[j][k])/pow(sigma,2) * (dldrk-dldrm[i][m]*costh) + 
                                                (pow(logLRT[j][m]-logLRT[j][k],2)/pow(sigma,3)-1./sigma)*dsigma_drik);
                            // The last factor above (second line) is the contribution from the derivative of sigma

                        } else {
                            double x0_m = x0meas[j][m][1];
                            double y0_m = y0meas[j][m][1];
                            double th_m = thmeas[j][m][1];
                            double ph_m = phmeas[j][m][1];
                            pp[k] += G;
                            double Rim = EffectiveDistance(xi,yi,x0_m,y0_m,th_m,ph_m,0);
                            if (Rim<Rmin) Rim = Rmin;

                            // To compute the cosine we need the point of closest distance to x[],y[] on the shower axis,
                            // for both k and m
                            // ------------------------------------------------------------------------------------------
                            double zcl_m = (cos(ph_m)*(xi-x0_m)+sin(ph_m)*(yi-y0_m))*(cos(th_m)/(1.+pow(cos(th_m),2)));
                            double xcl_m = x0_m+zcl_m*cos(th_m)*cos(ph_m);
                            double ycl_m = y0_m+zcl_m*cos(th_m)*sin(ph_m);
                            double costh = ((xi-xcl_m)*(xi-xcl_k) + (yi-ycl_m)*(yi-ycl_k) + zcl_k*zcl_m)/
                                            (Rik*Rim);
                            dpp_dRik[k] += G * ((logLRT[j][m]-logLRT[j][k])/pow(sigma,2) * (dldrk-dldrm[i][m]*costh) +
                                                (pow(logLRT[j][m]-logLRT[j][k],2)/pow(sigma,3)-1./sigma)*dsigma_drik);
                            // The last factor above (second line) is the contribution from the derivative of sigma
                        }
                    }
                    pg[k]  = pg[k] / (Fg*Nevents);      // Take into account how many we summed up
                    pp[k]  = pp[k] / ((1.-Fg)*Nevents); // 
                    if (pg[k]>0. && pg[k]>0.) inv_sigmafs += pow((pg[k]-pp[k])/(Fg*pg[k]+(1.-Fg)*pp[k]),2);
                    dpg_dRik[k] = dpg_dRik[k] / (Fg*Nevents);      // Because we summed Fg*Nevents of these
                    dpp_dRik[k] = dpp_dRik[k] / ((1.-Fg)*Nevents); // 
                }
                inv_sigmafs = sqrt(inv_sigmafs);
                if (inv_sigmafs==0.) inv_sigmafs = epsilon;

                // Now get variation of inverse sigma_fs over dR
                // ---------------------------------------------
                double sumx = 0.;
                double sumy = 0.;
                for (int k=Nevents; k<Nevents+Nbatch; k++) {
                    double den = pow(Fg * pg[k] + (1.-Fg)*pp[k],2);
                    double sqrden = sqrt(den);
                    double dif = pg[k]-pp[k];
                    if (dif==0. || den==0.) continue; // protect against adding nan to dudx
                    double d_invsigmafs_dRik = ( 2. * dpg_dRik[k] * (den*dif -pow(dif,2)*Fg*sqrden) + 
                                                    2. * dpp_dRik[k] * (-den*dif -pow(dif,2)*(1.-Fg)*sqrden)) / 
                                                    pow(den,2) / (2.*inv_sigmafs);
                    if (d_invsigmafs_dRik!=d_invsigmafs_dRik) continue;
                    double x0_k, y0_k, th_k, ph_k;
                    if (IsGamma[j][k]) {
                        x0_k = x0meas[j][k][0]; // reconstructed with the gamma hypothesis
                        y0_k = y0meas[j][k][0];
                        th_k = thmeas[j][k][0];
                        ph_k = phmeas[j][k][0];
                    } else {
                        x0_k = x0meas[j][k][1]; // reconstructed with the proton hypothesis
                        y0_k = y0meas[j][k][1];
                        th_k = thmeas[j][k][1];
                        ph_k = phmeas[j][k][1];
                    }
                    double Rik = EffectiveDistance(xi,yi,x0_k,y0_k,th_k,ph_k,0);
                    if (Rik<Rmin) Rik = Rmin;
                    double dR_dxi = EffectiveDistance(xi,yi,x0_k,y0_k,th_k,ph_k,1); // Partial derivative wrt x
                    double dR_dyi = EffectiveDistance(xi,yi,x0_k,y0_k,th_k,ph_k,2); // Partial derivative wrt y
                    sumx += d_invsigmafs_dRik * dR_dxi; 
                    sumy += d_invsigmafs_dRik * dR_dyi; 

                    // Accumulate the derivative of the utility with respect to x,y due to this energy bin
                    // -----------------------------------------------------------------------------------
                    dU_dxi += sumx * weight[j] * Fg;
                    dU_dyi += sumy * weight[j] * Fg;

                } 

            } // end j loop on energy points 

            // We apply a clamping of the derivatives to avoid divergent effects
            // -----------------------------------------------------------------
            dUdx->Fill(1.*epoch,fabs(dU_dxi));
            dUdx->Fill(1.*epoch,fabs(dU_dyi));

            /*
            if (fabs(dU_dxi)>5.) {
                dU_dxi = 5.*dU_dxi/fabs(dU_dxi); // a displacement of a det by 1m can't change U by more than 5!
            } 
            if (fabs(dU_dyi)>5.) {
                dU_dyi = 5.*dU_dyi/fabs(dU_dyi);
            } 
            */

            // If commonMode is 0, we directly update each detector position during the loop on i;
            // Otherwise we accumulate average displacements in radius or offset and fix them later
            // ------------------------------------------------------------------------------------
            if (commonMode==1) {

                // Now we know how the utility varies as a function of the distance of detector i from the showers,
                // measured in terms of the position of the detector x[], y[]. We use this information to vary the
                // detector position by taking all detectors at the same radius and averaging the derivative.
                // ------------------------------------------------------------------------------------------------
                double Ri; 
                int ir; 
                double dU_dRi;
                double d2 = pow(x[i],2)+pow(y[i],2);
                if (d2>0.) {
                    Ri = sqrt(d2);
                    double costheta = x[i]/(Ri+epsilon);
                    double sintheta = y[i]/(Ri+epsilon);
                    ir = (int)(Ri/(spanR+2*Rslack)*NRbins);
                    dU_dRi = dU_dxi*costheta + dU_dyi*sintheta;
                } else {
                    dU_dRi = sqrt(pow(dU_dxi,2)+pow(dU_dyi,2));
                    ir = 0;
                }
                if (ir<NRbins) {
                    displ[ir] += dU_dRi * LearningRateR[ir];
                    Ndispl[ir]++;
                } else {
                    cout << "     Warning, ir out of range" << endl;
                }
                if (debug) cout << "     i=" << i << " ir=" << ir << " Ndispl = " << Ndispl[ir] << " displ = " << displ[ir] << endl;

            } else if (commonMode==0) { // not vary R, vary independently x and y

                // Update independently each detector position based on gradient of U, ignoring 
                // the symmetry of the problem
                // ----------------------------------------------------------------------------
                double R = sqrt(pow(x[i],2)+pow(y[i],2));
                double costh = 0.;
                double sinth = 0.;
                double multiplier = LR_Scheduler(LearningRate[i],epoch);
                double dx = dU_dxi*multiplier;
                double dy = dU_dyi*multiplier;
                
                if (dx>maxDispl)  dx = maxDispl;
                if (dy>maxDispl)  dy = maxDispl;
                if (dx<-maxDispl) dx = -maxDispl;
                if (dy<-maxDispl) dy = -maxDispl;
                
                if (R>0.) {
                    costh = x[i]/R;
                    sinth = y[i]/R;
                }

                // Accumulate information on how consistent are the movements of detectors
                // -----------------------------------------------------------------------
                if (epoch>0) CosThetaEff = ((x[i]-xprev[i])*dx + (y[i]-yprev[i])*dy )/
                                           (sqrt(pow(x[i]-xprev[i],2)+pow(y[i]-yprev[i],2)+epsilon)*sqrt(pow(dx,2)+pow(dy,2)+epsilon));

                // Keep track of movement, so that we can update learning rate later
                // -----------------------------------------------------------------
                xprev[i] = x[i];
                yprev[i] = y[i];

                // Ok, now update the positions
                // ----------------------------
                x[i] = x[i] + dx;
                y[i] = y[i] + dy;
                double maxR = sqrt(pow(x[i],2)+pow(y[i],2));
                if (maxR>spanR+2*Rslack) {
                    x[i] = (spanR+2*Rslack-epsilon)*costh;
                    y[i] = (spanR+2*Rslack-epsilon)*sinth;
                }

                // Update learning rate based on "costhetaeff" value (see above) - this is the cosine
                // of the angle between a detector displacement and the previous detector displacement. If positive,
                // we increase the LR; if negative, we decrease it.
                // --------------------------------------------------------------------------------------------------
                double rate_modifier = CosThetaEff; // If using an average one it is better to instead use rate_modifier = -1.+2.*pow(0.5*(CosThetaEff+1.),2); // this is -1 for x=-1, +1 for x=1, and -0.3 for x=0
                LearningRate[i] *= exp(Momentum_coeff*rate_modifier);
                // Clamp them
                if (LearningRate[i]<0.01*StartLearningRate) LearningRate[i] = 0.01*StartLearningRate;
                if (LearningRate[i]>100.*StartLearningRate) LearningRate[i] = 100.*StartLearningRate;
                //CosDir->Fill(CosThetaEff);
                //CosvsEp->Fill(epoch,CosThetaEff);
                LR->Fill(epoch,log(LR_Scheduler(LearningRate[i],epoch)));
                aveDR += sqrt(pow(dx,2)+pow(dy,2));
                if (debug) cout << "aveDR " << aveDR  << " dx,dy " << dx << " " << dy << endl;
            } else if (commonMode==2) { // vary all coordinates jointly along common gradient
                commondx += dU_dxi;
                commondy += dU_dyi;
            } // end if commonMode

        } // end i loop on dets

        // For commonMode=1 or 2, we could not update the positions as we went, because we were accumulating
        // global increments. We do it now
        // -------------------------------------------------------------------------------------------------
        if (commonMode==1) { // vary R of detectors

            // Verify consistency of movements and modify learning rate for this radius
            // ------------------------------------------------------------------------
            for (int ir=0; ir<NRbins; ir++) {
                if (displ[ir]*prev_displ[ir]>0.) LearningRateR[ir] *= exp(Momentum_coeff); // This will apply to next iteration
            }

            // Compute average displacement as f(r)
            // ------------------------------------
            for (int ir=0; ir<NRbins; ir++) {
                prev_displ[ir] = displ[ir];
                if (Ndispl[ir]>0) displ[ir] = displ[ir]/Ndispl[ir];
                if (displ[ir]>maxDispl)  displ[ir] = maxDispl;
                if (displ[ir]<-maxDispl) displ[ir] = -maxDispl;
                if (debug) cout << ir << " " << Ndispl[ir] << " " << displ[ir] << endl;
            }

            // Now we have the required average displacement as a function of R and we apply to detectors
            // ------------------------------------------------------------------------------------------
            for (int i=0; i<Nunits; i++) {            
                double d2 = pow(x[i],2)+pow(y[i],2);
                double dx = 0.;
                double dy = 0.;
                double costh = 0.;
                double sinth = 0.;
                if (d2>0.) {
                    double R = sqrt(d2);
                    int ir = (int)(R/(spanR+2*Rslack)*NRbins);
                    costh = x[i]/(R+epsilon);
                    sinth = y[i]/(R+epsilon);
                    if (ir<NRbins) {
                        dx = costh * displ[ir];
                        dy = sinth * displ[ir];
                    }
                    x[i] = x[i] + dx;
                    y[i] = y[i] + dy;
                }
                double Rfinal = sqrt(pow(x[i],2)+pow(y[i],2));
                if (Rfinal>spanR+2*Rslack) {
                    x[i] = (spanR+2*Rslack-epsilon)*costh;
                    y[i] = (spanR+2*Rslack-epsilon)*sinth;
                }
                d2 = pow(dx,2)+pow(dy,2);
                if (d2>0.) aveDR += sqrt(d2);     
                if (debug) cout << "aveDR " << aveDR  << " dx,dy " << dx << " " << dy << endl;
            } // end i loop on dets
        } else if (commonMode==2) {
            cout << "commondx,dy = " << commondx << " " << commondy << endl;
            commondx = LearningRateC*commondx/Nunits;
            if (commondx>maxDispl) commondx = maxDispl;
            if (commondx<-maxDispl) commondx = -maxDispl;
            commondy = LearningRateC*commondy/Nunits;
            if (commondy>maxDispl) commondy = maxDispl;
            if (commondy<-maxDispl) commondy = -maxDispl;
            aveDR = sqrt(pow(commondx,2)+pow(commondy,2))*Nunits;
            for (int i=0; i<Nunits; i++) {
                x[i] += commondx;
                y[i] += commondy;
                if (x[i]>=spanR+2*Rslack)  x[i] = spanR+2*Rslack-epsilon;
                if (x[i]<=-spanR-2*Rslack) x[i] = -spanR-2*Rslack+epsilon;
                if (y[i]>=spanR+2*Rslack)  y[i] = spanR+2*Rslack-epsilon;
                if (y[i]<=-spanR-2*Rslack) y[i] = -spanR-2*Rslack+epsilon;
            }
        } // end if commonMode =1 or 2

        cout << "     Epoch = " << epoch << " Gf = " << GammaFraction[0] << "+-" << GammaFracErr[0] 
             << "  Utility value = " << Utility << "+-" << UtilityErr << " aveDR = " << aveDR/Nunits << endl;
        Layout->Reset();
        Layout2->Reset();
        Rdistr->Reset();
        for (int id=0; id<Nunits; id++) {
            Layout->Fill(x[id],y[id]);
            Layout2->Fill(x[id],y[id]);
            Rdistr->Fill(sqrt(x[id]*x[id]+y[id]*y[id]));
        }

        // Ensure the Rdistribution histograms stays visible
        // -------------------------------------------------
        int hmax = 0;
        for (int ib=0; ib<NbinsRdistr; ib++) {
            int h = Rdistr->GetBinContent(ib+1);
            if (h>hmax) hmax = h;
            h = Rdistr0->GetBinContent(ib+1);
            if (h>hmax) hmax = h;
        }
        Rdistr->SetMaximum(hmax*1.1);

        // Compute agreement metric
        // ------------------------
        double QP = pow(DXP->GetRMS(),2)+pow(DYP->GetRMS(),2)+pow(DXG->GetRMS(),2)+pow(DYG->GetRMS(),2);
        double QA = pow(DTHG->GetRMS(),2)+pow(DTHP->GetRMS(),2)+pow(DPHG->GetRMS(),2)+pow(DPHP->GetRMS(),2);
        double rmsx = (DXP->GetRMS()+DXG->GetRMS())*0.5;
        cout << "     Performance metric of shower position likelihood = " << sqrt(QP/4.) << " " << sqrt(QA/4.) << endl;
        PosQ->Fill(epoch,QP);
        AngQ->Fill(epoch,QA);

        CT = new TCanvas ("CT","",1300,600);
        CT->Divide(4,2);
        CT->cd(1);
        if (epoch>0) U->Fit("pol1","Q");
        U->SetMarkerStyle(20);
        U->SetLineWidth(3);
        U->SetMaximum(1.1*maxUtility);
        U->SetMinimum(0.);
        U->Draw("P");
        Uave->SetLineColor(kRed);
        Uave->Draw("SAME");
        CT->cd(2);
        Showers3->SetMarkerStyle(24);
        Showers3->SetMarkerSize(size);
        Showers3->Draw("COL4");
        Layout->Draw("SAME");
        CT->cd(3);
        Showers2->Draw();
        Layout2->Draw("SAME");
        CT->cd(4);
        Rdistr0->Draw();
        Rdistr->SetLineWidth(3);
        Rdistr->SetMinimum(0);
        Rdistr->Draw("SAME");
        CT->cd(5);
        PosQ->SetLineWidth(3);
        PosQ->Draw();
        CT->cd(6);
        AngQ->SetLineWidth(3);
        AngQ->Draw();
        CT->cd(7);
        LR->Draw();
        CT->cd(8);
        JSSum->Draw();
        //    CT->cd(7);
        //    CosDir->Draw();
        //    CT->cd(8);
        //    CosvsEp->SetMarkerStyle(20);
        //    CosvsEp->SetMarkerSize(size);
        //    CosvsEp->Draw();
        CT->Update();
        char namepng[40];
        sprintf (namepng,"./MODE/Layout%d.png",epoch+1);
        CT->Print(namepng);
        epoch++;


        // Adjust too small learning rates here
        // ------------------------------------
        //if (aveDR<maxDispl/10.) LearningRate = LearningRate*1.5;


        // Debug: check value of utility for same generated batch, after coordinates update
        // --------------------------------------------------------------------------------
        if (checkUtility) {
            for (int ie=0; ie<NEvalues; ie++) {
                for (int is=0; is<Nevents+Nbatch; is++) {
                    if (is%2==0) {
                        GenerateShower(ie,is,true);
                    } else {
                        GenerateShower(ie,is,false);
                    }
                    FindLogLR(ie,is); // Fills logLRT[] array and fills sigma2LRT
                } // end is loop
                cout << endl;
                GammaFraction[ie] = TrueGammaFraction[ie];
                GammaFracErr[ie]  = sqrt(VarianceGammaFraction(ie));      
            } // end ie loop
            double OldUtility = Utility;
            ComputeUtility(NEvalues); 
            cout << "New vs Old utility = " << Utility << " " << OldUtility << endl;
        }
    } while (epoch<Nepochs); // end SGD loop

    // Compute mutual distance graph 
    // -----------------------------
    for (int id=0; id<Nunits-1; id++) {
        for (int jd=id+1; jd<Nunits; jd++) {
            double r = sqrt(pow(x[id]-x[jd],2)+pow(y[id]-y[jd],2));
            MutualD->Fill(r);
        }
    }
    TCanvas * MD = new TCanvas ("MD","", 500,500);
    MD->cd();
    MutualD->Draw();


    // Plot histos of residuals in X0, Y0
    // ----------------------------------
    TCanvas * C1 = new TCanvas ("C1","",1200,500);
    C1->Divide(5,2);
    C1->cd(1);
    DXP->Draw();
    C1->cd(2);
    DYP->Draw();
    C1->cd(3);
    DXG->Draw();
    C1->cd(4);
    DYG->Draw();
    C1->cd(5);
    DTHP->Draw();
    C1->cd(6);
    DPHP->Draw();
    C1->cd(7);
    DTHG->Draw();
    C1->cd(8);
    DPHG->Draw();
    C1->cd(9);
    DTHPvsT->Draw("COL4");
    C1->cd(10);
    DTHGvsT->Draw("COL4");

    // Residuals for poisson variations of counts
    // ------------------------------------------
    TCanvas * Cr = new TCanvas ("Cr","",600,600);
    Cr->Divide(2,2);
    Cr->cd(1);
    DX0g->Draw();
    DX0p->SetLineColor(kRed);
    DX0p->Draw("SAME");
    Cr->cd(2);
    DY0g->Draw();
    DY0p->SetLineColor(kRed);
    DY0p->Draw("SAME");
    Cr->cd(3);
    DThg->Draw();
    DThp->SetLineColor(kRed);
    DThp->Draw("SAME");
    Cr->cd(4);
    DPhg->Draw();
    DPhp->SetLineColor(kRed);
    DPhp->Draw("SAME");
    

    TCanvas * C2 = new TCanvas ("C2","", 700, 500);
    C2->Divide(2,2);
    C2->cd(1);
    LLRP->SetLineWidth(3);
    LLRP->Draw();
    LLRG->SetLineWidth(3);
    LLRG->SetLineColor(kRed);
    LLRG->Draw("SAME");
    C2->cd(2);
    LLRPb->SetLineWidth(3);
    LLRPb->Draw();
    LLRGb->SetLineWidth(3);
    LLRGb->SetLineColor(kRed);
    LLRGb->Draw("SAME");
    C2->cd(3);
    SigLRT->Draw();
    C2->cd(4);
    SigLvsDR->Draw();

    // Compute agreement metric
    // ------------------------
    double QP = pow(DXP->GetRMS(),2)+pow(DYP->GetRMS(),2)+pow(DXG->GetRMS(),2)+pow(DYG->GetRMS(),2);
    double QA = pow(DTHG->GetRMS(),2)+pow(DTHP->GetRMS(),2)+pow(DPHG->GetRMS(),2)+pow(DPHP->GetRMS(),2);
    double QW = DXG->GetRMS();
    double QE = DYG->GetRMS();
    double QR = DXP->GetRMS();
    double QT = DYP->GetRMS();
    double QY = DTHG->GetRMS();
    double QU = DTHP->GetRMS();
    double QI = DPHG->GetRMS();
    double QO = DPHP->GetRMS();
    cout << "     Performance metric of shower position likelihood = " << sqrt(QP/4.) << " " << sqrt(QA/4.) << endl;
    cout << QW << " " << QE << " " << QR << " "<< QT << " " << QY << " " << QU << " "<< QI << " " << QO <<endl;

    // Plot results
    // ------------
    TCanvas * C = new TCanvas ("C","",1200,800);
    C->Divide(3,2);
    C->cd(1);
    U->SetLineWidth(3);
    U->SetMinimum(0.);
    U->Draw();
    Uave->SetLineColor(kRed);
    Uave->Draw("SAME");
    // C->cd(2);
    // GFmeas->SetMinimum(-0.2);
    // GFmeas->SetMaximum(1.2);
    // GFmeas->SetLineWidth(3);
    // GFmeas->SetLineColor(kRed);
    // GFmeas->Draw();
    // GFtrue->SetLineWidth(3);
    // GFtrue->Draw("SAME");
    C->cd(2);
    Layout->Draw("");
    Showers->Draw("SAME");
    C->cd(3);
    Rdistr->SetLineWidth(3);
    Rdistr->Draw();
    Rdistr0->Draw("SAME");
    C->cd(4);
    dUdx->Draw("PE");
    C->cd(5);
    PosQ->Draw("");
    C->cd(6);
    AngQ->Draw("");

    // Plot the distributions of fluxes per m^2
    // ----------------------------------------
    if (plotdistribs) {
        TH1D * MFG1 = new TH1D ("MFG1", "", 150, 0., 1500.);
        TH1D * EFG1 = new TH1D ("EFG1", "", 150, 0., 1500.);
        TH1D * MFP1 = new TH1D ("MFP1", "", 150, 0., 1500.);
        TH1D * EFP1 = new TH1D ("EFP1", "", 150, 0., 1500.);
        TH1D * MFG2 = new TH1D ("MFG2", "", 150, 0., 1500.);
        TH1D * EFG2 = new TH1D ("EFG2", "", 150, 0., 1500.);
        TH1D * MFP2 = new TH1D ("MFP2", "", 150, 0., 1500.);
        TH1D * EFP2 = new TH1D ("EFP2", "", 150, 0., 1500.);
        TH1D * MFG3 = new TH1D ("MFG3", "", 150, 0., 1500.);
        TH1D * EFG3 = new TH1D ("EFG3", "", 150, 0., 1500.);
        TH1D * MFP3 = new TH1D ("MFP3", "", 150, 0., 1500.);
        TH1D * EFP3 = new TH1D ("EFP3", "", 150, 0., 1500.);
        TH1D * MFG4 = new TH1D ("MFG4", "", 150, 0., 1500.);
        TH1D * EFG4 = new TH1D ("EFG4", "", 150, 0., 1500.);
        TH1D * MFP4 = new TH1D ("MFP4", "", 150, 0., 1500.);
        TH1D * EFP4 = new TH1D ("EFP4", "", 150, 0., 1500.);
        TH1D * MFG5 = new TH1D ("MFG5", "", 150, 0., 1500.);
        TH1D * EFG5 = new TH1D ("EFG5", "", 150, 0., 1500.);
        TH1D * MFP5 = new TH1D ("MFP5", "", 150, 0., 1500.);
        TH1D * EFP5 = new TH1D ("EFP5", "", 150, 0., 1500.);

        for (int i=0; i<150; i++) {
            double r = i*10.+0.5;
            MFG1->SetBinContent(i+1,MuFromG(0,r,0)/TankArea);
            EFG1->SetBinContent(i+1,EFromG(0,r,0)/TankArea);
            MFP1->SetBinContent(i+1,MuFromP(0,r,0)/TankArea);
            EFP1->SetBinContent(i+1,EFromP(0,r,0)/TankArea);
            MFG2->SetBinContent(i+1,MuFromG(1,r,0)/TankArea);
            EFG2->SetBinContent(i+1,EFromG(1,r,0)/TankArea);
            MFP2->SetBinContent(i+1,MuFromP(1,r,0)/TankArea);
            EFP2->SetBinContent(i+1,EFromP(1,r,0)/TankArea);
            MFG3->SetBinContent(i+1,MuFromG(2,r,0)/TankArea);
            EFG3->SetBinContent(i+1,EFromG(2,r,0)/TankArea);
            MFP3->SetBinContent(i+1,MuFromP(2,r,0)/TankArea);
            EFP3->SetBinContent(i+1,EFromP(2,r,0)/TankArea);
            MFG4->SetBinContent(i+1,MuFromG(3,r,0)/TankArea);
            EFG4->SetBinContent(i+1,EFromG(3,r,0)/TankArea);
            MFP4->SetBinContent(i+1,MuFromP(3,r,0)/TankArea);
            EFP4->SetBinContent(i+1,EFromP(3,r,0)/TankArea);
            MFG5->SetBinContent(i+1,MuFromG(4,r,0)/TankArea);
            EFG5->SetBinContent(i+1,EFromG(4,r,0)/TankArea);
            MFP5->SetBinContent(i+1,MuFromP(4,r,0)/TankArea);
            EFP5->SetBinContent(i+1,EFromP(4,r,0)/TankArea);
        }
        EFP1->SetMinimum(0.00000001);
        EFP1->SetMaximum(10000.);
        MFP1->SetMinimum(0.0000001);
        MFP1->SetMaximum(10.);
        EFG1->SetMinimum(0.00000001);
        EFG1->SetMaximum(10000.);
        MFG1->SetMinimum(0.00000001);
        MFG1->SetMaximum(1.);

        TCanvas * G = new TCanvas ("G","", 800, 800);
        G->Divide(2,2);
        G->cd(4);
        MFG1->Draw();
        MFG2->Draw("SAME");
        MFG3->Draw("SAME");
        MFG4->Draw("SAME");
        MFG5->Draw("SAME");
        G->cd(3);
        EFG1->Draw();
        EFG2->Draw("SAME");
        EFG3->Draw("SAME");
        EFG4->Draw("SAME");
        EFG5->Draw("SAME");
        G->cd(2);
        MFP1->Draw();
        MFP2->Draw("SAME");
        MFP3->Draw("SAME");
        MFP4->Draw("SAME");
        MFP5->Draw("SAME");
        G->cd(1);
        EFP1->Draw();
        EFP2->Draw("SAME");
        EFP3->Draw("SAME");
        EFP4->Draw("SAME");
        EFP5->Draw("SAME");
    }    

    TCanvas * DLC = new TCanvas ("DLC","",800,800);
    DLC->Divide(2,3);
    DLC->cd(1);
    DL->Draw("BOX");
    DLC->cd(2);
    StepsizeX->Draw("BOX");
    DLC->cd(3);
    StepsizeY->Draw("BOX");
    DLC->cd(4);
    StepsizeT->Draw("BOX");
    DLC->cd(5);
    StepsizeP->Draw("BOX");


    // If requested, write output geometry to file
    // -------------------------------------------
    if (writeGeom) SaveLayout();

    // End of program
    // --------------
    gROOT->Time();
    return;

}
