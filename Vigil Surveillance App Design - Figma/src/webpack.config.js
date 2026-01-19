const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const MiniCssExtractPlugin = require('mini-css-extract-plugin');

module.exports = (env, argv) => {
  const isDevelopment = argv.mode === 'development';

  return {
    entry: './src/index.js',
    output: {
      path: path.resolve(__dirname, 'dist'),
      filename: 'bundle.[contenthash].js',
      clean: true,
      publicPath: '/',
    },
    mode: isDevelopment ? 'development' : 'production',
    devtool: isDevelopment ? 'eval-source-map' : 'source-map',
    devServer: {
      static: {
        directory: path.join(__dirname, 'public'),
      },
      port: 3000,
      hot: true,
      historyApiFallback: true,
      open: true,
    },
    module: {
      rules: [
        {
          test: /\.(js|jsx)$/,
          exclude: /node_modules/,
          use: {
            loader: 'babel-loader',
            options: {
              presets: [
                '@babel/preset-env',
                ['@babel/preset-react', { runtime: 'automatic' }],
              ],
            },
          },
        },
        {
          test: /\.css$/,
          use: [
            isDevelopment ? 'style-loader' : MiniCssExtractPlugin.loader,
            'css-loader',
            'postcss-loader',
          ],
        },
        {
          test: /\.(png|svg|jpg|jpeg|gif|ico)$/i,
          type: 'asset/resource',
          generator: {
            filename: 'assets/images/[name].[hash][ext]',
          },
        },
        {
          test: /\.(woff|woff2|eot|ttf|otf)$/i,
          type: 'asset/resource',
          generator: {
            filename: 'assets/fonts/[name].[hash][ext]',
          },
        },
      ],
    },
    resolve: {
      extensions: ['.js', '.jsx'],
      alias: {
        '@': path.resolve(__dirname, 'src'),
        '@components': path.resolve(__dirname, 'src/components'),
        '@hooks': path.resolve(__dirname, 'src/hooks'),
        '@utils': path.resolve(__dirname, 'src/utils'),
        '@styles': path.resolve(__dirname, 'src/styles'),
      },
    },
    plugins: [
      new HtmlWebpackPlugin({
        template: './public/index.html',
        favicon: './public/favicon.ico',
      }),
      new MiniCssExtractPlugin({
        filename: 'styles.[contenthash].css',
      }),
    ],
    optimization: {
      splitChunks: {
        chunks: 'all',
        cacheGroups: {
          vendor: {
            test: /[\\/]node_modules[\\/]/,
            name: 'vendors',
            priority: 10,
          },
        },
      },
    },
  };
};
