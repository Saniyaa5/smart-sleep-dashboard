@echo off
REM ==========================================================
REM  Reset MySQL root password for Windows (MySQL 8.0+)
REM ==========================================================

echo.
echo ⚙️  Stopping MySQL service...
net stop mysql80
if errorlevel 1 net stop mysql

echo.
echo 🚀 Starting MySQL in skip-grant-tables mode...
cd /d "C:\Program Files\MySQL\MySQL Server 8.0\bin"
start cmd /k "mysqld --skip-grant-tables"
echo.
echo 🔒 Leave that new window open. When it finishes starting,
echo open a *second* Command Prompt (Run as Administrator)
echo and run:
echo ----------------------------------------------------------
echo mysql -u root
echo ----------------------------------------------------------
echo.
echo Then, inside the MySQL shell, paste these lines:
echo ----------------------------------------------------------
echo USE mysql;
echo ALTER USER 'root'@'localhost' IDENTIFIED BY '1234';
echo FLUSH PRIVILEGES;
echo EXIT;
echo ----------------------------------------------------------
echo.
echo ✅ After that, return here and press any key to stop MySQL
pause

echo.
echo 🛑 Stopping temporary MySQL server...
taskkill /F /IM mysqld.exe >nul 2>&1

echo.
echo ▶️  Starting MySQL service normally...
net start mysql80
if errorlevel 1 net start mysql

echo.
echo ✅ Root password reset complete!
echo Login with:  mysql -u root -p  (password: 1234)
pause
