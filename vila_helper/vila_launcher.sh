gnome-terminal ./vila_scripts/controller.sh
sleep 5

gnome-terminal ./vila_scripts/web_page.sh
sleep 5

gnome-terminal ./vila_scripts/model.sh $1
sleep 10

firefox http://0.0.0.0:7860