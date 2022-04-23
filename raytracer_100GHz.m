%For reproducibility
rng(7);
%Viewing Topographic map of Chicago
%viewer = siteviewer("Buildings","chicago.osm","Basemap","topographic");

%Defining base station (transmitter) sites
%%tx properties
names = ["S Clarke"];
lats = [41.878313];
lons = [-87.630772];
heights = [6];
freq = 100e9;
transmit_power = 10;
%%Antenna properties
%TODO Change Antenna Properties
antenna_size = [64, 1];
antenna_spacing = [0.0015, 0.0015];
cfgArray = arrayConfig("Size",antenna_size,"ElementSpacing",antenna_spacing);
%%Creating tx
tx = txsite("Name",names,"Latitude",lats,"Longitude",lons,"TransmitterFrequency",freq,"Antenna",cfgArray,"AntennaHeight",heights,"TransmitterPower",transmit_power);
%disp(tx(1,1));
%show(tx);

% %Defining receiver locations
% %Cluster 1 at Millenium Park : 1000 receivers
% n_recv=1000;
% lat_min = 41.880990;
% lat_max = 41.884139;
% lon_min = -87.624050;
% lon_max = -87.621128;
% names = string(1:n_recv);
% lats = lat_min + (lat_max-lat_min)*rand(1,n_recv);
% lons = lon_min + (lon_max-lon_min)*rand(1,n_recv);
% %%Antenna properties
% %TODO Change Antenna Properties
% antenna_size = [1, 1];
% antenna_spacing = [0.0015, 0.0015];
% cfgArray = arrayConfig("Size",antenna_size,"ElementSpacing",antenna_spacing);
% rx1 = rxsite("Name",names,"Latitude",lats,"Longitude",lons,"Antenna",cfgArray);
% show(rx1);
%Cluster 2 on Clark St : 1000 receivers
n_recv=100000;
lat_min = 41.878381;
lat_max = 41.880677;
lon_min = -87.630803;
lon_max = -87.630818;
names = string(1:n_recv);
lats = lat_min + (lat_max-lat_min)*rand(1,n_recv);
lons = lon_min + (lon_max-lon_min)*rand(1,n_recv);
height = 2;
%%Antenna properties
%TODO Change Antenna Properties
antenna_size = [1, 1];
antenna_spacing = [0.0015, 0.0015];
cfgArray = arrayConfig("Size",antenna_size,"ElementSpacing",antenna_spacing);
rx1 = rxsite("Name",names,"Latitude",lats,"Longitude",lons,"Antenna",cfgArray,"AntennaHeight",height);
% show(rx1);
% %Cluster 3 on Chicago Ave : 1000 receivers
% n_recv=1000;
% lat_min = 41.896638;
% lat_max = 41.896797;
% lon_min = -87.630828;
% lon_max = -87.622428;
% names = string(2*n_recv+1:3*n_recv);
% lats = lat_min + (lat_max-lat_min)*rand(1,n_recv);
% lons = lon_min + (lon_max-lon_min)*rand(1,n_recv);
% %%Antenna properties
% %TODO Change Antenna Properties
% antenna_size = [1, 1];
% antenna_spacing = [0.0015, 0.0015];
% cfgArray = arrayConfig("Size",antenna_size,"ElementSpacing",antenna_spacing);
% rx3 = rxsite("Name",names,"Latitude",lats,"Longitude",lons,"Antenna",cfgArray);
% show(rx3);

%Ray Tracing Model combined with atmospheric (gas) attenuation
%TODO Change the properties of rptm and raytrace (Building material concrete etc)
rtpm = propagationModel("raytracing",Method="sbr",MaxNumReflections=0);
gas_pm = propagationModel("gas");
%fog_pm = propagationModel("fog");
rtPlusWeather = rtpm + gas_pm;
%clearMap(viewer);
%raytrace(tx,rx1,rtPlusWeather);
%raytrace(tx,rx2,rtPlusWeather);
%raytrace(tx,rx3,rtPlusWeather);
rays1 = raytrace(tx,rx1,rtPlusWeather);
%rays2 = raytrace(tx,rx2,rtPlusWeather);
%rays3 = raytrace(tx,rx3,rtPlusWeather);

%Creating Datafiles for Ingestion
%%Location data of transmitters and receivers

%%CIR File (Contains phase in degrees, ToA in seconds, and received power in dbm for each path)
CIR = [n_recv];
DoD = [n_recv];
DoA = [n_recv];
LOS = [n_recv];
curr_paths = 0;
curr = 1;
for i = 1:n_recv
    num_paths_max = 1;
    num_paths = 0;
    CIR = [CIR i 0];
    DoD = [DoD i 0];
    DoA = [DoA i 0];
    curr = curr + 2;
    while num_paths<num_paths_max
%         fprintf ("%d %d %d %d\n", 1, i, 1, num_paths+1);
%         fprintf ("%f %f %f\n", rays1{1,i}(1,num_paths+1).PhaseShift, rays1{1,i}(1,num_paths+1).PropagationDelay, rays1{1,i}(1,num_paths+1).PathLoss);
        try
            %LOS Status only for first path
            if num_paths==0
                LOS_status = rays1{1,i}(1,num_paths+1).LineOfSight;
                LOS = [LOS LOS_status];
            end
            phase = 180 / pi * rays1{1,i}(1,num_paths+1).PhaseShift;
            ToA = rays1{1,i}(1,num_paths+1).PropagationDelay;
            recv_power = -1 * (rays1{1,i}(1,num_paths+1).PathLoss + 30);
            departure_azi = rays1{1,i}(1,num_paths+1).AngleOfDeparture(1,1);
            departure_ele = rays1{1,i}(1,num_paths+1).AngleOfDeparture(2,1);
            arrival_azi = rays1{1,i}(1,num_paths+1).AngleOfArrival(1,1);
            arrival_ele = rays1{1,i}(1,num_paths+1).AngleOfArrival(2,1);
            num_paths = num_paths + 1;
            CIR = [CIR num_paths phase ToA recv_power];
            DoD = [DoD num_paths departure_azi departure_ele 9999];
            DoA = [DoA num_paths arrival_azi arrival_ele 9999];
            %fprintf ('Happening')
        catch
            %fprintf ('Done for');
            %If broke at 0 paths then put LOS_status = -1
            if num_paths == 0
                LOS = [LOS -1];
            end
            break
        end
    end
    CIR(1,curr) = num_paths;
    DoD(1,curr) = num_paths;
    DoA(1,curr) = num_paths;
    curr = curr + 4 * num_paths;
end

%Converting arrays to .mat files for ingestion by deepMIMO generator
if ~exist('./Raytracing_scenarios', 'dir')
    fprintf("Creating Raytracing Scenarios directory\n")
    mkdir './Raytracing_scenarios'
end
if exist('./Raytracing_scenarios/O1_100', 'dir')
    fprintf("Deleting old O1_100 directory\n")
    rmdir './Raytracing_scenarios/O1_100' s
end
fprintf("Creating new O1_100 directory\n")
mkdir './Raytracing_scenarios/O1_100'
save('./Raytracing_scenarios/O1_100/O1_100.1.CIR.mat', 'CIR');
save('./Raytracing_scenarios/O1_100/O1_100.1.DoD.mat', 'DoD');
save('./Raytracing_scenarios/O1_100/O1_100.1.DoA.mat', 'DoA');
save('./Raytracing_scenarios/O1_100/O1_100.1.LoS.mat', 'LOS');