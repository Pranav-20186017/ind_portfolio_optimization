import React from "react";
import {
  Typography,
  Container,
  Box,
  Paper,
  Grid,
  Link as MuiLink,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Button,
  Card,
  CardContent,
  Divider
} from "@mui/material";
import Head from "next/head";
import Link from "next/link";
import "katex/dist/katex.min.css";
import { BlockMath, InlineMath } from "react-katex";
import TopNav from "../../components/TopNav";

/* -------------------------------------------------------------------
   EQUATION COMPONENT                                                    
   ------------------------------------------------------------------- */
const Equation: React.FC<{ math: string }> = ({ math }) => (
  <Box sx={{ p: 2, bgcolor: "#f5f5f5", borderRadius: 1, my: 2, textAlign: "center" }}>
    <BlockMath math={math} />
  </Box>
);

/* -------------------------------------------------------------------
   GARCH BETA PAGE                                                      
   ------------------------------------------------------------------- */
const GARCHBetaPage: React.FC = () => {
  return (
    <>
      <Head>
        <title>GARCH Beta | Portfolio Optimisation</title>
        <meta
          name="description"
          content="A time-varying market sensitivity measure that accounts for volatility clustering, providing more accurate risk estimates during periods of market turbulence."
        />
      </Head>

      <TopNav />

      <Container maxWidth="lg" sx={{ py: 6 }}>
        {/* NAVIGATION */}
        <Box sx={{ mb: 4, display: 'flex', gap: 2 }}>
          <Link href="/docs" passHref>
            <Button variant="outlined" color="primary">← Back to Docs</Button>
          </Link>
          <Link href="/" passHref>
            <Button variant="outlined" color="secondary">← Back to Portfolio Optimizer</Button>
          </Link>
        </Box>

        {/* HEADER */}
        <Box sx={{ mb: 6, textAlign: 'center' }}>
          <Typography variant="h3" component="h1" gutterBottom>
            GARCH Beta
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            A dynamic beta estimation technique that incorporates time-varying volatility to capture evolving market sensitivity.
          </Typography>
        </Box>

        {/* INTUITION */}
        <Paper elevation={3} sx={{ p: 3, my: 4 }}>
          <Typography variant="h5" gutterBottom>
            Why GARCH Beta?
          </Typography>
          <Typography paragraph>
            Traditional beta assumes constant volatility across time, but financial markets exhibit volatility clustering—periods 
            of high volatility tend to be followed by more high volatility, and calm periods tend to persist as well. 
            <strong> GARCH Beta</strong> addresses this limitation by incorporating time-varying volatility and covariance 
            estimates from Generalized Autoregressive Conditional Heteroskedasticity (GARCH) models.
          </Typography>
          <Typography paragraph>
            This approach recognizes that an asset's relationship with the market is not static but evolves with changing 
            market conditions, providing more accurate risk estimates during volatile periods and improving portfolio 
            risk management during market turbulence.
          </Typography>
        </Paper>

        {/* MATHEMATICS */}
        <Typography variant="h4" gutterBottom>
          1. Mathematical Definition
        </Typography>
        <Typography variant="h6" gutterBottom>
          1.1 The Problem with Constant Beta
        </Typography>
        <Typography paragraph>
          The traditional beta coefficient is defined as:
        </Typography>
        <Equation math="\beta = \frac{\operatorname{Cov}(r_i,\,r_m)}{\operatorname{Var}(r_m)}" />
        <Typography paragraph>
          However, this standard approach assumes that covariance and variance are constant over time, which contradicts the 
          observed volatility dynamics in financial markets. During market stress, correlations and volatilities tend to increase, 
          potentially causing beta to change significantly.
        </Typography>

        <Typography variant="h6" gutterBottom>
          1.2 GARCH Process Foundations
        </Typography>
        <Typography paragraph>
          A GARCH(1,1) model for the variance of a return series is specified as:
        </Typography>
        <Equation math="\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2" />
        <Typography paragraph>
          Where <InlineMath math="\sigma_t^2" /> is the conditional variance at time <InlineMath math="t" />, 
          <InlineMath math="\epsilon_{t-1}" /> is the previous period's return shock, and <InlineMath math="\omega" />, 
          <InlineMath math="\alpha" />, and <InlineMath math="\beta" /> are parameters that determine how quickly volatility 
          responds to market shocks and how persistent it is.
        </Typography>

        <Typography variant="h6" gutterBottom>
          1.3 Multivariate GARCH for Beta Estimation
        </Typography>
        <Typography paragraph>
          To estimate time-varying beta, we need a bivariate GARCH model that captures the dynamic relationship between asset and market returns:
        </Typography>
        <Equation math="\begin{pmatrix} r_{i,t} \\ r_{m,t} \end{pmatrix} = \begin{pmatrix} \mu_i \\ \mu_m \end{pmatrix} + \begin{pmatrix} \epsilon_{i,t} \\ \epsilon_{m,t} \end{pmatrix}" />
        <Typography paragraph>
          Where <InlineMath math="r_{i,t}" /> and <InlineMath math="r_{m,t}" /> are the returns for the asset and market at time <InlineMath math="t" />, 
          <InlineMath math="\mu_i" /> and <InlineMath math="\mu_m" /> are their respective expected returns, and the error terms 
          <InlineMath math="\epsilon_{i,t}" /> and <InlineMath math="\epsilon_{m,t}" /> have a time-varying covariance matrix <InlineMath math="H_t" />:
        </Typography>
        <Equation math="H_t = \begin{pmatrix} h_{ii,t} & h_{im,t} \\ h_{im,t} & h_{mm,t} \end{pmatrix}" />
        <Typography paragraph>
          The time-varying GARCH beta at time <InlineMath math="t" /> is then calculated as:
        </Typography>
        <Equation math="\beta_t = \frac{h_{im,t}}{h_{mm,t}}" />
        <Typography paragraph>
          Where <InlineMath math="h_{im,t}" /> is the conditional covariance between asset and market returns, and <InlineMath math="h_{mm,t}" /> 
          is the conditional variance of market returns.
        </Typography>

        <Typography variant="h6" gutterBottom>
          1.4 Common GARCH-Family Models for Beta
        </Typography>
        <Typography paragraph>
          Several multivariate GARCH specifications are used for beta estimation:
        </Typography>
        <ul>
          <li>
            <Typography paragraph>
              <strong>DCC-GARCH (Dynamic Conditional Correlation):</strong> Decomposes the covariance matrix into conditional standard 
              deviations and correlations, allowing for direct modeling of time-varying correlations.
            </Typography>
          </li>
          <li>
            <Typography paragraph>
              <strong>BEKK-GARCH:</strong> Ensures positive definiteness of the covariance matrix by modeling it directly, 
              capturing spillover effects between asset and market volatilities.
            </Typography>
          </li>
          <li>
            <Typography paragraph>
              <strong>GO-GARCH (Generalized Orthogonal):</strong> Uses orthogonal transformations to simplify the estimation process.
            </Typography>
          </li>
        </ul>

        <Typography variant="h6" gutterBottom>
          1.5 Financial Significance
        </Typography>
        <Typography paragraph>
          GARCH Beta captures several important financial phenomena:
        </Typography>
        <ol>
          <li>
            <Typography paragraph>
              <strong>Volatility clustering:</strong> Periods of high market volatility, which often coincide with market crises, 
              are properly accounted for in risk estimates.
            </Typography>
          </li>
          <li>
            <Typography paragraph>
              <strong>Dynamic correlations:</strong> Asset-market correlations often increase during downturns, a phenomenon that 
              GARCH Beta can capture but static beta cannot.
            </Typography>
          </li>
          <li>
            <Typography paragraph>
              <strong>Regime shifts:</strong> Gradual or sudden changes in market conditions are reflected in evolving beta estimates.
            </Typography>
          </li>
        </ol>

        {/* DEFAULT PARAMS */}
        <Typography variant="h4" gutterBottom>
          2. Default Parameters
        </Typography>
        <TableContainer component={Paper} sx={{ mb: 4 }}>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Parameter</TableCell>
                <TableCell>Default</TableCell>
                <TableCell>Description</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {[
                ["model", "\"DCC\"", "GARCH model specification (DCC, BEKK, GO)"],
                ["p", "1", "GARCH lag order for conditional variance"],
                ["q", "1", "ARCH lag order for squared innovations"],
                ["window", "500", "Estimation window in trading days (~2 years)"],
                ["distribution", "\"normal\"", "Distribution assumption (normal, student-t, skewed-t)"],
                ["rf", "0.0", "Risk-free rate subtracted from returns before estimation"],
              ].map(([param, def, desc]) => (
                <TableRow key={param as string}>
                  <TableCell>{param}</TableCell>
                  <TableCell>
                    <code>{def as string}</code>
                  </TableCell>
                  <TableCell>{desc}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>

        {/* IMPLEMENTATION CONSIDERATIONS */}
        <Typography variant="h4" gutterBottom>
          3. Implementation Considerations
        </Typography>
        <Typography paragraph>
          When implementing GARCH Beta in portfolio optimization:
        </Typography>
        <ul>
          <li>
            <Typography paragraph>
              <strong>Computational complexity:</strong> GARCH models, especially multivariate ones, can be computationally intensive to estimate. 
              DCC-GARCH offers a reasonable balance between accuracy and computational efficiency.
            </Typography>
          </li>
          <li>
            <Typography paragraph>
              <strong>Sample size requirements:</strong> GARCH models require substantial historical data (typically 500+ observations) 
              for reliable parameter estimation.
            </Typography>
          </li>
          <li>
            <Typography paragraph>
              <strong>Distribution assumptions:</strong> Financial returns often exhibit fat tails and skewness. Using Student's t or 
              skewed-t distributions can improve model fit compared to normal distribution assumptions.
            </Typography>
          </li>
          <li>
            <Typography paragraph>
              <strong>Forecasting horizon:</strong> GARCH models excel at short-term volatility forecasting but may be less accurate 
              for long-term predictions.
            </Typography>
          </li>
          <li>
            <Typography paragraph>
              <strong>Stability:</strong> Parameter estimation can be sensitive to outliers and initial conditions. 
              It's advisable to use robust optimization methods and check for parameter stability.
            </Typography>
          </li>
        </ul>

        {/* PROS / CONS */}
        <Typography variant="h4" gutterBottom>
          4. Advantages and Limitations
        </Typography>
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
              <Typography variant="h6" gutterBottom color="primary">
                Advantages
              </Typography>
              <ul>
                <li>Captures time-varying risk dynamics that static beta ignores.</li>
                <li>Accounts for volatility clustering and changing correlations during market stress.</li>
                <li>Provides more accurate risk forecasts during turbulent market periods.</li>
                <li>Allows for regime-specific risk management strategies.</li>
                <li>Can significantly improve Value-at-Risk and Expected Shortfall estimation.</li>
              </ul>
            </Box>
          </Grid>
          <Grid item xs={12} md={6}>
            <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
              <Typography variant="h6" gutterBottom color="error">
                Limitations
              </Typography>
              <ul>
                <li>Significantly more complex to implement than static beta models.</li>
                <li>Requires substantial historical data for reliable parameter estimation.</li>
                <li>Model specification choices (GARCH type, orders, distribution) add subjectivity.</li>
                <li>Computationally intensive, especially for large portfolios.</li>
                <li>Parameter estimates can be unstable or converge to boundaries.</li>
              </ul>
            </Box>
          </Grid>
        </Grid>

        {/* REFERENCES */}
        <Paper elevation={2} sx={{ p: 4, mb: 4, mt: 4 }}>
          <Typography variant="h4" gutterBottom>
            5. References
          </Typography>
          <ul>
            <li>Bollerslev, T. (1986). <em>Generalized Autoregressive Conditional Heteroskedasticity</em>. <em>Journal of Econometrics</em>, 31(3), 307-327.</li>
            <li>Engle, R. F. (2002). <em>Dynamic Conditional Correlation: A Simple Class of Multivariate GARCH Models</em>. <em>Journal of Business & Economic Statistics</em>, 20(3), 339-350.</li>
            <li>Engle, R. F., & Kroner, K. F. (1995). <em>Multivariate Simultaneous Generalized ARCH</em>. <em>Econometric Theory</em>, 11(1), 122-150.</li>
            <li>Bauwens, L., Laurent, S., & Rombouts, J. V. (2006). <em>Multivariate GARCH Models: A Survey</em>. <em>Journal of Applied Econometrics</em>, 21(1), 79-109.</li>
            <li>Caporin, M., & McAleer, M. (2013). <em>Ten Things You Should Know About the Dynamic Conditional Correlation Representation</em>. <em>Econometrics</em>, 1(1), 115-126.</li>
            <li>Andersen, T. G., Bollerslev, T., Christoffersen, P. F., & Diebold, F. X. (2006). <em>Volatility and Correlation Forecasting</em>. <em>Handbook of Economic Forecasting</em>, 1, 777-878.</li>
          </ul>
        </Paper>

        {/* RELATED TOPICS */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h4" gutterBottom>
            Related Topics
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6} md={3}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Welch Beta
                </Typography>
                <Typography variant="body2" paragraph>
                  A robust alternative to traditional beta that uses winsorization to reduce the impact of extreme returns.
                </Typography>
                <Link href="/docs/welch-beta" passHref>
                  <Button variant="contained" color="primary">
                    Learn More
                  </Button>
                </Link>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Semi Beta
                </Typography>
                <Typography variant="body2" paragraph>
                  A downside beta that measures portfolio sensitivity to the benchmark only during down markets.
                </Typography>
                <Link href="/docs/semi-beta" passHref>
                  <Button variant="contained" color="primary">
                    Learn More
                  </Button>
                </Link>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Rolling Beta
                </Typography>
                <Typography variant="body2" paragraph>
                  A time-series analysis of beta that shows how an asset's relationship with the market changes over time.
                </Typography>
                <Link href="/docs/rolling-beta" passHref>
                  <Button variant="contained" color="primary">
                    Learn More
                  </Button>
                </Link>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Volatility
                </Typography>
                <Typography variant="body2" paragraph>
                  A statistical measure of the dispersion of returns, usually measured using standard deviation.
                </Typography>
                <Link href="/docs/volatility" passHref>
                  <Button variant="contained" color="primary">
                    Learn More
                  </Button>
                </Link>
              </Box>
            </Grid>
          </Grid>
        </Paper>
      </Container>
    </>
  );
};

export default GARCHBetaPage; 