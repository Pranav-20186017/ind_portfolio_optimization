# ind_portfolio_optimization

Building Portfolio Optimizer in the Indian Market

# MOSEK License Setup

## Local Development

To use MOSEK in local development:

1. Place your `mosek.lic` file in one of these locations:

   - `./mosek/mosek.lic` (in the project root)
   - `./mosek.lic` (in the project root)
   - `~/mosek/mosek.lic` (in your home directory)

2. Alternatively, set one of these environment variables:
   - `MOSEK_LICENSE_PATH`: Full path to your license file
   - `MOSEK_LICENSE_CONTENT`: Base64-encoded content of your license file

## CI/CD and Docker Deployment

For CI/CD and Docker deployment, the license is handled securely:

1. Add your MOSEK license as a GitHub secret named `MOSEK_LICENSE_CONTENT`:

   - Encode your license file as base64: `base64 -w 0 mosek.lic > mosek.lic.base64`
   - Add the content of `mosek.lic.base64` as a secret in your repository settings

2. The workflow automatically passes this to:

   - Tests during CI/CD runs
   - Docker build process

3. The application will automatically detect and use the license at runtime.
